import torch
import time
import psutil
import os
import gc
from tqdm import tqdm

def clear_pytorch_caches():
    """Clear PyTorch internal caches safely"""
    gc.collect()
    
    # Try different methods to clear PyTorch caches
    if hasattr(torch, 'jit') and hasattr(torch.jit, '_state'):
        try:
            torch.jit._state._python_cu.clear_cache()
        except AttributeError:
            pass
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection again
    gc.collect()

#model = torch.jit.load("./exports/Distil_XLSR_5_Custom_Trans_Layer_ConformerTCM_conf-1-2_best_mdt_lora_mini_wrapper_no_pad3mobile.pt")

# qat
model = torch.jit.load("./exports/best_avg_5_best_mdt_lora_mini_wrapper_from_MDT_241214_lora_250501_hotfixmobile.pt")

input = torch.randn(1, 33600)

def benchmark_inference(model, input, runs=300, num_threads=None):
    # Set number of CPU threads if specified
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads")
    
    times = []
    process = psutil.Process(os.getpid())
    mem_usages = []
    
    # Warm up run to stabilize memory
    with torch.no_grad():
        _ = model(input)
        clear_pytorch_caches()  # Clear caches after warmup
    
    with torch.no_grad():
        for i in tqdm(range(runs)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # Clear GPU cache
                start_mem = torch.cuda.memory_allocated()
                start = time.time()
                _ = model(input.cuda())
                torch.cuda.synchronize()
                end = time.time()
                end_mem = torch.cuda.memory_allocated()
                mem_usages.append(end_mem - start_mem)
            else:
                # Clear caches before each run
                clear_pytorch_caches()
                start_mem = process.memory_info().rss
                start = time.time()
                _ = model(input)
                end = time.time()
                end_mem = process.memory_info().rss
                mem_usages.append(end_mem - start_mem)
            
            times.append(end - start)
            
            # Print memory usage every 50 runs for debugging
            if (i + 1) % 50 == 0:
                current_mem = process.memory_info().rss / 1024 / 1024  # MB
                print(f"Run {i+1}: Current memory usage: {current_mem:.2f} MB")
    
    avg_time = sum(times) / len(times)
    peak_time = max(times)
    fastest_time = min(times)
    peak_mem = max(mem_usages)
    avg_mem = sum(mem_usages) / len(mem_usages)
    return {
        'average_time': avg_time,
        'peak_time': peak_time,
        'fastest_time': fastest_time,
        'average_memory_usage': avg_mem,
        'peak_memory_usage': peak_mem
    }

if __name__ == "__main__":
    # Use more CPU cores for better performance
    NUM_CPU_CORES = 1  # Changed from 1 to 4
    
    if torch.cuda.is_available():
        model = model.cuda()
        input = input.cuda()
    
    # Set memory-efficient settings for CPU
    torch.set_num_interop_threads(1)  # Reduce inter-op parallelism
    torch.set_num_threads(NUM_CPU_CORES)
    
    results = benchmark_inference(model, input, num_threads=NUM_CPU_CORES)
    print("Benchmark results:")
    for k, v in results.items():
        print(f"{k}: {v}")
    
    # Final memory cleanup
    clear_pytorch_caches()
