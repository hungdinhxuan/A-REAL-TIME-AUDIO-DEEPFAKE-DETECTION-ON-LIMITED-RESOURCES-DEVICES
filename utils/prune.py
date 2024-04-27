from student import *
from teacher import *
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
from torch.cuda.amp import autocast
import torch.nn.utils.prune as prune
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
os.environ['TORCH_MOBILE_INFERENCE'] = '0'

device = "cuda" if torch.cuda.is_available() else "cpu"
from menu import get_main_menu
from models import SSLHuggingFaceModel
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining, AutoModelForPreTraining
from utils import *
from torchaudio.models.wav2vec2.utils import import_fairseq_model, import_huggingface_model
from startup_config import set_random_seed

args = get_main_menu()
set_random_seed(1, args)



class W2V2_TA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        feat, _ = self.model(x)
        print("new forward")
        return feat


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x: Tensor):
        wav_padded = pad(x).unsqueeze(0)
        output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = self.model(wav_padded)

        # Softmax the output and get the probability of fake in tensor
        # output = nn.Softmax(dim=1)(output)[:,0].item()
        output = self.softmax(output)[:,0]
        
        return output

class WrapperHalfModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x: Tensor):
        
        wav_padded = pad(x).unsqueeze(0)
        
        
        output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = self.model(wav_padded)

        # Softmax the output and get the probability of fake in tensor
        # output = nn.Softmax(dim=1)(output)[:,0].item()
        # output = self.softmax(output)[:,0]
        
        return output
input = torch.randn(1, 64600).to(device)
checkpoint = '/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_40_1e-06_self_KD_teacher_W2VBase/best_checkpoint_42.pth'

model = SelfDistil_W2V2BASE_AASISTL(device)
model = nn.DataParallel(model).to(device)
# model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)

# # Pruning
# model = model.module

# module = model.attention[0]
# print(list(module.named_parameters()))
for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    # if isinstance(module, torch.nn.Conv2d):
    #     prune.l1_unstructured(module, name='weight', amount=0.2)
    #     print("Prune Conv2d")
    # # prune 40% of connections in all linear layers
    # elif isinstance(module, torch.nn.Linear):
    #     prune.l1_unstructured(module, name='weight', amount=0.4)
    #     print("Prune Linear")
    
    try:
        prune.l1_unstructured(module, name='weight', amount=0.9)
        print(f"Prune {module}")
        prune.remove(module, 'weight')
    except:
        pass
    
# save the pruned model with buffer only
torch.save(model.state_dict(), "Distil_W2V2BASE_AASISTL_Self_KD_pruned.pt")