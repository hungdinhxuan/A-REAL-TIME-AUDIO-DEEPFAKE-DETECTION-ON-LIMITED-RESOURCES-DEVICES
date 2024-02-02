import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from teacher import W2V2_AASIST
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor, Distil_W2V2BASE_AASISTL, Distil_W2V2BASE_AASISTL_Cosine, Distil_W2V2BASE_AASISTL_Regressor, Distil_W2V2FTBASE_AASISTL, Distil_W2V2BASE_AASISTL_Self_KD, Distil_W2V2BASE_AASISTL_Self_KD_Teacher
import numpy as np
from torch import Tensor
import librosa

import logging

# Get the Numba logger
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)  # Set level to WARNING, ERROR, or CRITICAL


os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"


model = Distil_W2V2BASE_AASISTL_Self_KD(device=device)
model = nn.DataParallel(model).to(device)

model_path = "/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_64_1e-06_self_KD/best_checkpoint_46.pth"

# Load the model
model.load_state_dict(torch.load(model_path,map_location=device))
print("Loaded model from {}".format(model_path))

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
    
    def forward(self, x: Tensor):
     
        wav_padded = pad(x).unsqueeze(0)
        output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = self.model(wav_padded)
        print("output 1",output)

        print("Softmax: ", nn.Softmax(dim=1)(output))
        # Softmax the output
        output = nn.Softmax(dim=1)(output)[:,0]
        
        print("output 2",output)

        # Final result in % fake
        return output
audio, sr = librosa.load("/datab/hungdx/KDW2V-AASISTL/MMSTTS_ara_000008.wav",sr=16000)
# Inference
input = torch.randn(1, 36000)


model = WrapperModel(model.module).to(device)

model.eval()

with torch.no_grad():
    # convert audio to tensor
    audio = torch.from_numpy(audio).to(device)
    print(model(audio))

    # # Scripting module for mobile deployment
    scripted_model = torch.jit.script(model, input)
#     # scripted_model.save("Distil_W2V2BASE_AASISTL_Self_KD.pt")
