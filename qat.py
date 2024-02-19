from student import *
from teacher import *
from torch import Tensor
import torch
import torch.nn as nn
import librosa
import os
from torch.utils.mobile_optimizer import optimize_for_mobile


from menu import get_main_menu
# from data_utils import pad
from startup_config import set_random_seed
from main import W2V2_TA
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def pad(x, max_len: int = 64600) -> torch.Tensor:
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
        wav_padded = pad(x).unsqueeze(0)
        # output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = self.model(wav_padded)
        output = self.model(wav_padded)
        # Return the probability of being spoofed
        return self.softmax(output)[0][0]
    
# Load spoofed sample
input,_ = librosa.load("LA_T_1541806.wav", sr=16000)
input = torch.tensor(input).unsqueeze(0)
print(input.shape)
padded_input = pad(input).unsqueeze(0)

# checkpoint = '/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_3_v10/best_checkpoint_63.pth'
checkpoint = '/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth'

# model = SelfDistil_W2V2BASE_AASISTL(device)
model = W2V2_AASIST(device)

model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))

print("Before replace")

model.eval()

with torch.no_grad():
    
    # before,spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(padded_input)
    before = model(padded_input)
    print(before)

    # Replace frontend
    
    model.module.ssl_model = W2V2_TA(import_fairseq_model(
        model.module.ssl_model.model
    )).to(device)

print("After replace")

model.eval()

with torch.no_grad():
    # after,spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(padded_input)
    after = model(padded_input)
    print(after)

# torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)

# Scriptable
model_fp32 = WrapperModel(model.module).to(device)
model_fp32.eval()

# Save model
jit_model = torch.jit.script(model_fp32)
print("After script")
# Optimize for mobile
opt_model = optimize_for_mobile(jit_model)
print("After optimize_for_mobile")
with torch.no_grad():
    after = opt_model(input)
    print(after)

opt_model.save("W2V2-AASIST-teacher_scripted.pt")

jit_model = torch.jit.load("W2V2-AASIST-teacher_scripted.pt")
with torch.no_grad():
    jit_out = jit_model(input)
    print("JIT model output")
    print(jit_out)

# Save
# opt_model._save_for_lite_interpreter("W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_best_checkpoint_126.ptl")
print("Done~")
