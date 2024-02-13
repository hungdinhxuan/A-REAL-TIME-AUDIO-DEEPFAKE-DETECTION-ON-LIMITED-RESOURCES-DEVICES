from student import Distil_W2V2BASEHG_AASISTL_Self_KD, Distil_W2V2BASE_AASISTL_Self_KD_Teacher, Distil_W2V2BASEHG_AASISTL_Self_KD_Teacher, Distil_W2V2BASE_AASISTL_Self_KD, Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from data_utils import *
from student import *
# Quantization code
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from torch.utils.data import DataLoader
from neural_compressor.utils.pytorch import load
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
from models import SSLModelBase
from argparse import Namespace
from startup_config import set_random_seed

args = Namespace(database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22', protocols_path='protocol.txt', batch_size=25, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/nfs/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=False, KD_cosine=False, KD_mse=False, self_KD=False, self_KD_teacher=True, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40, workers=8)

set_random_seed(args.seed, args)

train_loader, dev_loader = get_train_dev_dataloader(args, "rawboost", dataset="cnsl")

vector_space = []
device = 'cpu' 
student_model = SelfDistil_W2V2BASE_AASISTL(device).to(device)

for i, (x, y) in enumerate(train_loader):
    vector = student_model.ssl_model(x.to(device))
    # Save the vector space to a file
    torch.save(vector, f'vector_{i}.pt')


