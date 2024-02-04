from student import Distil_W2V2BASEHG_AASISTL_Self_KD, Distil_W2V2BASE_AASISTL_Self_KD_Teacher, Distil_W2V2BASEHG_AASISTL_Self_KD_Teacher, Distil_W2V2BASE_AASISTL_Self_KD, Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher
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
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
from models import SSLModelBase

model = SSLModelBase(device)

model.frozen()

audio = torch.randn(1, 64600).to(device)
model.eval()

print(model(audio))

model.unfrozen()

print(model(audio))