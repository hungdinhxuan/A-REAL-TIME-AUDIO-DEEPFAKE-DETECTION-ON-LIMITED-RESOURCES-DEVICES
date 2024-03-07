from student import *
from teacher import *
from torch import Tensor
import torch
import torch.nn as nn
import librosa
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Optional

from menu import get_main_menu
# from data_utils import pad
from startup_config import set_random_seed
from main import W2V2_TA
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PADDING_SIZE = 64600


def pad(x, max_len: int = PADDING_SIZE):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(f"Using device {device}")

args = get_main_menu()
set_random_seed(1221, args)


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Padding with 2s
        wav_padded = pad(x, 64600).unsqueeze(0)
        # output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = self.model(
        #     wav_padded)
        output = self.model(wav_padded)
        print(output)
        # Return the probability of being spoofed
        return self.softmax(output)[0][0]


# Load spoofed sample
input, _ = librosa.load("LA_T_1541806.wav", sr=16000)
input = torch.tensor(input).unsqueeze(0)
print(input.shape)
padded_input = pad(input).unsqueeze(0)

checkpoint = '/nfs/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_4_v14/best_checkpoint_35.pth'
# checkpoint = '/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth'

model = SelfDistil_W2V2BASE_AASISTL(
    device, ssl_cpkt_path='/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt')
# model = W2V2_AASIST(device)

model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))

print("Before replace")

model.eval()

with torch.no_grad():

    before, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(
        padded_input)
    # before = model(padded_input)
    print(before)

    # Replace frontend

    model.module.ssl_model = W2V2_TA(import_fairseq_model(
        model.module.ssl_model.model
    )).to(device)

print("After replace")

model.eval()

with torch.no_grad():
    after, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(
        padded_input)
    # after = model(padded_input)
    print(after)

# torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)


# Load another model
model = Distil_W2V2BASE_AASISTL(
    device, ssl_cpkt_path='/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt')

model = nn.DataParallel(model).to(device)

# Load checkpoint and replace frontend

# Strict = False because the model is not the same (Distil_W2V2BASE_AASISTL is not the same as SelfDistil_W2V2BASE_AASISTL)
# SelfDistil_W2V2BASE_AASISTL have some more Linear layer (which using for self KD training not for inference)
model.load_state_dict(torch.load(
    checkpoint, map_location=device), strict=False)

model.module.ssl_model = W2V2_TA(import_fairseq_model(
    model.module.ssl_model.model
)).to(device)

model.eval()

print("After replace with Distil_W2V2BASE_AASISTL")
with torch.no_grad():
    after = model(padded_input)
    print(after)

# Scriptable
model_fp32 = WrapperModel(model.module).to(device)
model_fp32.eval()

SAVE_MODEL_PATH = "W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_4_v14_best35_no_optimize_mobile_bf16.pt"
with torch.cpu.amp.autocast():
    model_fp32 = torch.jit.script(model_fp32)
    model_fp32 = torch.jit.freeze(model_fp32)
    y = model_fp32(padded_input)
    print("After autotrace")
    print(y)
    optimized_for_inference = torch.jit.optimize_for_inference(model_fp32)

    torch.jit.save(
        optimized_for_inference, SAVE_MODEL_PATH)

# # Half precision
# model_fp16 = WrapperModel(model.module.half()).to(device)

# # Save model
# jit_model = torch.jit.script(model_fp16)
# print("After script")

# # # Not optimize for mobile and save


# print("Saving model to ", SAVE_MODEL_PATH)

# optimized_for_inference = torch.jit.optimize_for_inference(jit_model)

# torch.jit.save(
#     optimized_for_inference, SAVE_MODEL_PATH)


# Optimize for mobile
# opt_model = optimize_for_mobile(jit_model)
# print("After optimize_for_mobile")
# with torch.no_grad():
#     after = opt_model(input)
#     print(after)

# # Save optimized model
# print("Saving optimized model to ",
#       "W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_4_v14_best35_mobile.pt")
# opt_model.save(
#     "W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_4_v14_best35_mobile.pt")

# jit_model = torch.jit.load(
#     SAVE_MODEL_PATH)
# with torch.no_grad():
#     jit_out = jit_model(input)
#     print("JIT model output")
#     print(jit_out)

# Save
# opt_model._save_for_lite_interpreter("W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_best_checkpoint_126.ptl")
print("Done~")
