import torch
import os
import librosa
import numpy as np
from menu import get_main_menu
import librosa
# from data_utils import pad
from startup_config import set_random_seed
import fairseq
from models import Custom_Wav2Vec2_Fe, My_XLSR_FE
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PreTrainedModel
from torchinfo import summary
from torchdistill.core.forward_hook import ForwardHookManager
from main import W2V2_TA
import torch.nn.functional as F
from torch import nn

# Define a linear layer to transform from 256 to 1024 dimensions


class LowRankTransform(nn.Module):
    def __init__(self, in_features, out_features):
        super(LowRankTransform, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MSELoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(LowRankTransform, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.linear(x), y)


class ConvExpand(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(ConvExpand, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Expecting x of shape [batch_size, sequence_length, in_features]
        # Change to [batch_size, in_channels, sequence_length] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Change back to [batch_size, sequence_length, out_features]
        x = x.permute(0, 2, 1)
        return x


args = get_main_menu()
set_random_seed(1221, args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = Custom_Wav2Vec2_Fe(device).to(device)
my_xlsr = My_XLSR_FE(device).to(device)
print(my_xlsr)

# my_xlsr = W2V2_TA(import_fairseq_model(my_xlsr.model)).to(device)

# model_hook_manager = ForwardHookManager(device)
# my_xlsr_forward_hook_manager = ForwardHookManager(device)

# model_hook_manager.add_hook(
#     model, 'model.encoder.transformer.layers.0.final_layer_norm', requires_input=True, requires_output=True)

# my_xlsr_forward_hook_manager.add_hook(
#     my_xlsr, 'model.encoder.transformer.layers.0.final_layer_norm', requires_input=True, requires_output=True)

# dummy_input = torch.randn(3, 16000).to(device)
# # summary(model, input_size=(1, 16000), depth=5)
# model(dummy_input)
# my_xlsr(dummy_input)

# io_model_dict = model_hook_manager.pop_io_dict()
# io_my_xlsr_dict = my_xlsr_forward_hook_manager.pop_io_dict()
# # Create an instance of the LowRankTransform
# transform = LowRankTransform(in_features=256, out_features=1024).to(device)

# # torch.Size([1, 49, 256])
# model_out = io_model_dict['model.encoder.transformer.layers.0.final_layer_norm']['output']
# # torch.Size([1, 49, 1024])
# my_xlsr_out = io_my_xlsr_dict['model.encoder.transformer.layers.0.final_layer_norm']['output']

# # Linear interpolation to make the shape of the output of the model and my_xlsr the same
# model_out_interpolated = F.interpolate(
#     model_out, size=my_xlsr_out.shape[2], mode='linear', align_corners=True)

# conv_expand = ConvExpand(256, 1024).to(device)

# # Transform my_xlsr_out to match the dimensions of model_out
# model_out_expand = conv_expand(model_out)
# model_out_transformed = transform(model_out)
# print(model_out_expand.shape)

# print(model)
# print(my_xlsr)
# summary(model, input_size=(1, 16000), depth=5)
# summary(my_xlsr, input_size=(1, 16000), depth=3)


# print(model(dummy_input).shape)
# print(my_xlsr(dummy_input).shape)

# summary(model, input_size=(1, 16000))

# feature_extractor = Wav2Vec2FeatureExtractor(
#     feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

# model = Wav2Vec2PreTrainedModel.from_pretrained(
#     "facebook/wav2vec2-base-960h", feature_extractor=feature_extractor)

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

# model_file = '/datad/hungdx/KDW2V-AASISTL/pretrained/xlsr2_300m.pt'
# model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([
#                                                                     model_file])
# original = model[0]

# original.encoder.layers = original.encoder.layers[:6]
# imported = import_fairseq_model(original)

# # print number of parameters

# # print("Original model: ", sum(p.numel() for p in original.parameters()))
# print("Imported model: ", sum(p.numel() for p in imported.parameters()))
