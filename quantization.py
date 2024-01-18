import sys
import os

import torch
from torch import nn
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor, Distil_W2V2BASE_AASISTL_Cosine
import numpy as np
from torch import Tensor

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"

model = Distil_W2V2BASE_AASISTL_Cosine(device=device)
model = nn.DataParallel(model).to(device)

# model_path = "/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_30_1e-06_W2V2Base_KD_cosine/best_checkpoint_42.pth"

# # Load the model
# model.load_state_dict(torch.load(model_path,map_location=device))
# print("Loaded model from {}".format(model_path))

# Define a wrapper model
class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @torch.jit.script
    def pad(self, x, max_len: int = 64600) -> Tensor:
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
        return padded_x
        
    @torch.jit.script
    def forward(self, x: Tensor):
        wav_padded = self.pad(x).unsqueeze(0)
        output, regressor_output = self.model(wav_padded)

        # Softmax the output and get the probability of fake in tensor
        # output = nn.Softmax(dim=1)(output)[:,0].item()
        
        return output

# Inference
input = torch.randn(1, 16000).to(device)
model = WrapperModel(model).to(device)

with torch.inference_mode():
    print(model(input))

# torch.save(model.state_dict(), "W2V2Base_KD_cosine_best_checkpoint_42.pt")
# print("Saved model to W2V2Base_KD_cosine_best_checkpoint_42.pt")
# model = WrapperModel(model)

model.load_state_dict(torch.load("W2V2Base_KD_cosine_best_checkpoint_42.pt"))

print("Tracing model...")
# Trace the model
traced_model = torch.jit.trace(model, input)
traced_model.save("W2V2Base_KD_cosine_traced_best_checkpoint_42.pt")
print("Traced model saved to W2V2Base_KD_cosine_traced_best_checkpoint_42.pt")
