import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from teacher import W2V2_AASIST
import numpy as np
from torch import Tensor
import librosa

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda"

model = W2V2_AASIST()
model = nn.DataParallel(model).to(device)

model_path = "/nfs/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth"

# Load the model
model.load_state_dict(torch.load(model_path,map_location=device))
print("Loaded model from {}".format(model_path))

# Define a wrapper model
class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def pad(self, x, max_len: int = 64600) -> Tensor:
        x_len = torch.tensor(x.shape[0])
        max_len = torch.tensor(max_len)

        if torch.ge(x_len, max_len).item():
            return x[:max_len]
        # need to pad
        num_repeats = int((max_len / x_len).ceil().item())
        
        padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
        return padded_x
    
    def forward(self, x: Tensor):
        wav_padded = self.pad(x).unsqueeze(0)
        output = self.model(wav_padded)

        

        # Softmax the output
        output = nn.Softmax(dim=1)(output)[:,0].item()
        print("output 2",output)

        # Final result in % fake
        return output
audio, sr = librosa.load("/datab/hungdx/KDW2V-AASISTL/MMSTTS_kor_000046.wav",sr=16000)
# Inference
input = torch.randn(1, 16000)


model = WrapperModel(model).to(device)

with torch.inference_mode():
    # convert audio to tensor
    audio = torch.from_numpy(audio).to(device)
    print(model(audio))

# Scripting module for mobile deployment
# scripted_model = torch.jit.trace(model, input)
# scripted_model.save("W2V2-AASIST-teacher2.pt")