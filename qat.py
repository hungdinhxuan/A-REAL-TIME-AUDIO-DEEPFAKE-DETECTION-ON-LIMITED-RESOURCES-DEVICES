from student import *
from teacher import *
from torch import Tensor
import torch
import torch.nn as nn
import os
from torch.utils.mobile_optimizer import optimize_for_mobile


from menu import get_main_menu
from utils import *
from startup_config import set_random_seed
from main import W2V2_TA
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(f"Using device {device}")

args = get_main_menu()
set_random_seed(args.seed, args)


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x: Tensor):
        wav_padded = pad(x).unsqueeze(0)
        output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = self.model(wav_padded)
        output = self.softmax(output)[:,0]
        return output

input = torch.randn(1, 64600).to(device)
checkpoint = '/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher/best_checkpoint_126.pth'

model = SelfDistil_W2V2BASE_AASISTL(device)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))

print("Before replace")

model.eval()

with torch.no_grad():
    
    before,spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(input)
    print(before)

    # Replace frontend
    
    model.module.ssl_model = W2V2_TA(import_fairseq_model(
        model.module.ssl_model.model
    )).to(device)

print("After replace")

with torch.no_grad():
    after,spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(input)
    print(after)

# torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)

# Scriptable
model_fp32 = WrapperModel(model.module).to(device)
model_fp32.eval()

# Dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
# jit_model = torch.jit.script(model_int8)
# print("After script")
# opt_model = optimize_for_mobile(jit_model)
# print("After optimize_for_mobile")

# opt_model._save_for_lite_interpreter("W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_best_checkpoint_126_quant_int8.ptl")
# print("Done~")

# Save model
jit_model = torch.jit.script(model_fp32)
print("After script")
# Optimize for mobile
opt_model = optimize_for_mobile(jit_model)
print("After optimize_for_mobile")
with torch.no_grad():
    after = opt_model(input)
    print(after)

# opt_model.save("W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_best_checkpoint_126.pt")

# Save
opt_model._save_for_lite_interpreter("W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_best_checkpoint_126.ptl")
print("Done~")
