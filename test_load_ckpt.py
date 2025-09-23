import torch
orig_torch_load = torch.load
import logging
from collections import OrderedDict
def torch_wrapper(*args, **kwargs):
    logging.warning("[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper


ckpt = torch.load("/nvme1/hungdx/logs/train/runs/2025-09-23_00-49-50/checkpoints/epoch_012.ckpt")

state_dict = ckpt["state_dict"]

print(state_dict.keys())
# remove prefix 'net.'
new_state_dict = OrderedDict()

    # Process the state dict keys
required_prefix = 'net'
for key, value in state_dict.items():
    # Skip keys that don't start with the required prefix
    if not key.startswith(required_prefix):
        continue

    # Remove the 'net.' prefix to match model's state dict keys
    new_key = key[len(required_prefix) + 1:]  # +1 for the dot after prefix
    new_state_dict[new_key] = value
#print(ckpt["state_dict"].keys())

print(new_state_dict.keys())
# save to pth
torch.save(new_state_dict, "pretrained/wav2vec2_5_conformertcm_pretrained_stage1_epoch_012.pth")
print("Saved to pretrained/wav2vec2_5_conformertcm_pretrained_stage1_epoch_012.pth")