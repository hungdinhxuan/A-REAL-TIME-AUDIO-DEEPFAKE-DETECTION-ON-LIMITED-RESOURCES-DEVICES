from student import Distil_W2V2BASEHG_AASISTL_Self_KD
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
# Quantization code
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from torch.utils.data import DataLoader
from neural_compressor.utils.pytorch import load
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
from menu import get_main_menu

args = get_main_menu()


model = Distil_W2V2BASEHG_AASISTL_Self_KD(device)
model = torch.nn.DataParallel(model).to(device)

model_path = "/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_80_1e-06_self_KD_W2VBaseHG/best_checkpoint_37.pth"

# Load the model
# model.load_state_dict(torch.load(model_path,map_location=device))
# print("Loaded model from {}".format(model_path))
# model.eval()
@torch.jit.script
def pad(x, max_len: int = 64600) -> Tensor:
    x_len = torch.tensor(x.shape[0])
    max_len = torch.tensor(max_len)

    if torch.ge(x_len, max_len).item():
        return x[:max_len]
        # need to pad
    num_repeats = int((max_len / x_len).ceil().item())
        
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x
# Define a wrapper model
class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        wav_padded = pad(x).unsqueeze(0) if x.dim() == 1 else x
        output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = self.model(wav_padded)
        return output

# Inference test
input = torch.randn(1, 16000).to(device)

model = WrapperModel(model.module).to(device)
model.eval()

# Evaluate the model
with torch.no_grad():
    print("Inference...")
    output = model(input)
    print(output)
    print("Done inference.")