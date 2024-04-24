import torch
import os
import librosa
import numpy as np
from menu import get_main_menu
import librosa
# from data_utils import pad
from startup_config import set_random_seed
import fairseq
from torchaudio.models.wav2vec2.utils import import_fairseq_model


args = get_main_menu()
set_random_seed(1221, args)

# # Load Scaled the model
# model_scaled = "/nfs/datab/hungdx/KDW2V-AASISTL/exports/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best_checkpoint_11_scaled2mobile.pt"
# input, _ = librosa.load("LA_T_1541806.wav", sr=16000)
# input = torch.tensor(input).unsqueeze(0)

# jit_model = torch.jit.load(model_scaled)

# with torch.no_grad():
#     print("Scaled model")
#     print(jit_model(input))

# # Load the model
# model_name = "/nfs/datab/hungdx/KDW2V-AASISTL/exports/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best_checkpoint_11_mobile.pt"

# jit_model = torch.jit.load(model_name)
# with torch.no_grad():
#     print("Normal model")
#     print(jit_model(input))

model_file = '/datad/hungdx/KDW2V-AASISTL/pretrained/xlsr2_300m.pt'
model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                    model_file])
original = model[0]

original.encoder.layers = original.encoder.layers[:6]
imported = import_fairseq_model(original)

# print number of parameters

# print("Original model: ", sum(p.numel() for p in original.parameters()))
print("Imported model: ", sum(p.numel() for p in imported.parameters()))
