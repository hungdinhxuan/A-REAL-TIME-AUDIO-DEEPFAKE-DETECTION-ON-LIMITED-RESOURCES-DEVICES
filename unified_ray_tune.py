import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import time
import yaml
import re
from typing import Any, Dict, Optional, Tuple

# Ray Tune imports (Ray 1.13.0 compatible)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
try:
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.suggest.basic_variant import BasicVariantGenerator
except Exception:
    ConcurrencyLimiter = None
    BasicVariantGenerator = None


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = config
    for i, p in enumerate(parts):
        is_last = i == len(parts) - 1
        # Determine if next container should be list or dict when creating
        next_is_index = False
        if not is_last:
            nxt = parts[i + 1]
            next_is_index = nxt.isdigit()

        if isinstance(cur, list):
            if not p.isdigit():
                raise KeyError(f"Trying to index list with non-integer key: {p} in {dotted_key}")
            idx = int(p)
            # Ensure list is long enough
            while len(cur) <= idx:
                cur.append([] if next_is_index else {})
            if is_last:
                cur[idx] = value
                return
            cur = cur[idx]
        elif isinstance(cur, dict):
            if is_last:
                cur[p] = value
                return
            if p not in cur or not isinstance(cur[p], (dict, list)):
                cur[p] = [] if next_is_index else {}
            cur = cur[p]
        else:
            raise TypeError(f"Unsupported container type at key '{p}' while setting '{dotted_key}'")


def _apply_overrides(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    for k, v in overrides.items():
        _set_nested(cfg, k, v)
    return cfg


def _parse_metrics_from_line(line: str) -> Tuple[Optional[float], Optional[float]]:
    # Returns (eval_accuracy, dev_loss) if present in this line, else (None, None)
    acc = None
    dev = None
    # Example: [VALIDATION] eval_accuracy:  72.345
    m_acc = re.search(r"\[VALIDATION\]\s+eval_accuracy:\s*([0-9.]+)", line)
    if m_acc:
        try:
            acc = float(m_acc.group(1))
        except Exception:
            acc = None

    # Example logger line: 'Epoch: X - train_loss: A - dev_loss: B'
    m_dev = re.search(r"dev_loss:\s*([0-9.]+)", line)
    if m_dev:
        try:
            dev = float(m_dev.group(1))
        except Exception:
            dev = None

    return acc, dev


def trainable_fn(ray_cfg: Dict[str, Any]):
    base_yaml: str = ray_cfg["base_yaml"]
    database_path: str = "/home/hungdx/code/A-REAL-TIME-AUDIO-DEEPFAKE-DETECTION-ON-LIMITED-RESOURCES-DEVICES/data/KD25/" # fix
    protocols_path: str = "/home/hungdx/code/A-REAL-TIME-AUDIO-DEEPFAKE-DETECTION-ON-LIMITED-RESOURCES-DEVICES/data/KD25/protocol.txt" # fix
    python_bin: str = ray_cfg.get("python_bin", sys.executable)
    cuda_visible_devices: Optional[str] = ray_cfg.get("cuda_visible_devices")

    # Load base YAML and apply overrides (excluding reserved keys)
    base_cfg = _read_yaml(base_yaml)
    overrides = {k: v for k, v in ray_cfg.items() if k not in {
        "base_yaml", "database_path", "protocols_path", "python_bin", "cuda_visible_devices"
    }}

    # Unique trial name
    # Try Ray 1.x way to get trial id; fallback to env/timestamp
    trial_id_getter = getattr(tune, "get_trial_id", None)
    if callable(trial_id_getter):
        trial_id = trial_id_getter()
    else:
        trial_id = os.getenv("TUNE_TRIAL_ID", str(int(time.time())))
    base_name = base_cfg.get("name", os.path.splitext(os.path.basename(base_yaml))[0])
    trial_name = f"{base_name}__{trial_id}"

    # Ensure per-trial model/log directories are unique by overriding config.name
    overrides["name"] = trial_name

    trial_cfg = _apply_overrides(base_cfg, overrides)

    # Create a temporary YAML for this trial
    workdir = tempfile.mkdtemp(prefix=f"raytune_{trial_id}_")
    trial_yaml = os.path.join(workdir, "config.yaml")
    _write_yaml(trial_cfg, trial_yaml)

    # Build subprocess command
    cmd = [
        python_bin,
        os.path.join(os.path.dirname(__file__), "unified_main.py"),
        "--yaml", trial_yaml,
        f"--database_path={database_path}",
        f"--protocols_path={protocols_path}",
    ]

    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Safer NCCL defaults for single-GPU Ray trials to avoid NCCL invalid usage
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Keep threads modest to reduce contention
    env.setdefault("OMP_NUM_THREADS", "4")

    # Stream stdout and parse metrics as they appear
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(__file__),
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    last_acc: Optional[float] = None
    last_dev: Optional[float] = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)

            acc, dev = _parse_metrics_from_line(line)
            if acc is not None:
                last_acc = acc
            if dev is not None:
                last_dev = dev

            # Report when we have at least one of them
            if (last_acc is not None) or (last_dev is not None):
                tune.report(
                    eval_accuracy=last_acc if last_acc is not None else float("nan"),
                    dev_loss=last_dev if last_dev is not None else float("nan"),
                )

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Training process exited with code {ret}")

    finally:
        # Keep artifacts; but remove temp yaml dir to avoid buildup
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


def build_search_space(preset: str) -> Dict[str, Any]:
    # Keys are dotted paths into the YAML config
    if preset == "quick":
        return {
            #"train.learning_rate": tune.loguniform(1e-6, 5e-4),
            "train.alpha": tune.uniform(0.1, 1),
            "unified_loss.loss.weights.cosine": tune.uniform(0.5, 1.5),
            "unified_loss.loss.weights.mse": tune.loguniform(1e-5, 1),
            "unified_loss.pooling.method": tune.choice(["mean", "weighted_sum", "weighted_mean", "sum", "max", "min", "last"]),
            "criterions.0.kwargs.beta": tune.uniform(0.1, 1),
            "criterions.0.kwargs.temperature": tune.randint(1, 5),
            "unified_loss.projection.type": tune.choice(["linear", "shallow_ae", "mlp", "sequential_deep_ae", "null"]),
            "unified_loss.projection.target": tune.choice(["teacher", "student", "null"]),
            "unified_loss.projection.kwargs.use_bias": tune.choice([True, False]),
            "unified_loss.projection.kwargs.dims": tune.choice([[1024, 768, 768], [1024, 1024, 768]]),
            "unified_loss.projection.kwargs.hidden_dim": tune.choice([1024, 768]),
            "unified_loss.projection.input_dim": tune.choice([1024, 768]),
            "unified_loss.projection.output_dim": tune.choice([1024, 768]),
            "unified_loss.loss.enabled": tune.choice(["mse", "cosine", "recon"]),
            "unified_loss.loss.weights.recon": tune.loguniform(1e-5, 1),
            "unified_loss.loss.weights.l1": tune.loguniform(1e-5, 1),

        }
    elif preset == "lr_alpha":
        return {
            "train.learning_rate": tune.loguniform(1e-6, 1e-4),
            "train.alpha": tune.uniform(0.2, 0.6),
        }
    else:
        # Default balanced preset
        return {
            "train.learning_rate": tune.loguniform(5e-6, 5e-4),
            "train.alpha": tune.uniform(0.2, 0.6),
            "unified_loss.loss.weights.cosine": tune.uniform(0.5, 1.2),
            "unified_loss.loss.weights.mse": tune.loguniform(5e-5, 5e-4),
            "unified_loss.pooling.method": tune.choice(["mean", "weighted_sum"]),
        }


def main():
    parser = argparse.ArgumentParser(description="Ray Tune runner for unified_main.py")
    parser.add_argument("--base_yaml", required=True, help="Path to base YAML config")
    # parser.add_argument("--database_path", required=True, help="Dataset root path (as used by unified_main.py)")
    # parser.add_argument("--protocols_path", required=True, help="Protocols file path (as used by unified_main.py)")
    parser.add_argument("--preset", default="default", choices=["quick", "lr_alpha", "default"], help="Search space preset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials")
    parser.add_argument("--max_concurrent_trials", type=int, default=2, help="Max concurrent trials")
    parser.add_argument("--gpus_per_trial", type=float, default=1.0, help="GPUs per trial (Ray resource)")
    parser.add_argument("--cpus_per_trial", type=float, default=4.0, help="CPUs per trial (Ray resource)")
    parser.add_argument("--cuda_visible_devices", default=None, help="Optional CUDA_VISIBLE_DEVICES to set for trials")
    parser.add_argument("--python_bin", default=sys.executable, help="Python binary to launch unified_main.py")
    args = parser.parse_args()

    space = build_search_space(args.preset)

    # Wrap trainable to inject fixed args
    def wrapped_trainable(cfg: Dict[str, Any]):
        cfg = dict(cfg)
        cfg.update({
            "base_yaml": os.path.abspath(args.base_yaml),
            "database_path": "/home/hungdx/code/A-REAL-TIME-AUDIO-DEEPFAKE-DETECTION-ON-LIMITED-RESOURCES-DEVICES/data/KD25/",
            "protocols_path": "/home/hungdx/code/A-REAL-TIME-AUDIO-DEEPFAKE-DETECTION-ON-LIMITED-RESOURCES-DEVICES/data/KD25/protocol.txt",
            "python_bin": args.python_bin,
        })
        if args.cuda_visible_devices:
            cfg["cuda_visible_devices"] = args.cuda_visible_devices
        trainable_fn(cfg)

    # Define scheduler (Ray 1.13.0)
    scheduler = ASHAScheduler(
        max_t=10000,
        grace_period=1,
        reduction_factor=2,
    )

    # Concurrency control (Ray 1.13.0): use BasicVariantGenerator(max_concurrent=...)
    search_alg = None
    if BasicVariantGenerator is not None and args.max_concurrent_trials:
        search_alg = BasicVariantGenerator(max_concurrent=args.max_concurrent_trials)

    analysis = tune.run(
        wrapped_trainable,
        config=space,
        metric="dev_loss",
        mode="min",
        num_samples=args.num_samples,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        scheduler=scheduler,
        search_alg=search_alg,
    )

    best_trial = analysis.get_best_trial(metric="dev_loss", mode="min", scope="all")
    if best_trial is not None:
        print("Best dev_loss:", best_trial.last_result.get("dev_loss"))
        print("Best eval_accuracy:", best_trial.last_result.get("eval_accuracy"))
        print("Best config:")
        best_cfg = {k: v for k, v in best_trial.config.items() if k not in {"base_yaml", "database_path", "protocols_path", "python_bin", "cuda_visible_devices"}}
        for k, v in best_cfg.items():
            print(f"  {k}: {v}")
    else:
        print("No successful trials found.")


if __name__ == "main":
    main()

if __name__ == "__main__":
    main()


