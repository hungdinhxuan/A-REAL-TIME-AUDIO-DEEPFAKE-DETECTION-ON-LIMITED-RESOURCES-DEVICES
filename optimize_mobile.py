from student import *
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from torch import nn
import os
from main import W2V2_TA
from torchaudio.models.wav2vec2.utils import import_fairseq_model

device = "cuda" if torch.cuda.is_available() else "cpu"
cp_path = '/nfs/datab/hungdx/KDW2V-AASISTL/distilXLSR_xlsr128.pt'   # Change the pre-trained XLSR model path. 
checkpoint = torch.load(cp_path, map_location=device)
pretrained_model_cfg = checkpoint["Config"]["model"]
pretrained_model_cfg = DistilXLSRConfig(pretrained_model_cfg)
model = DistilXLSR(pretrained_model_cfg)
model.load_state_dict(checkpoint["Student"], strict=False)

torch.jit.script(model)
