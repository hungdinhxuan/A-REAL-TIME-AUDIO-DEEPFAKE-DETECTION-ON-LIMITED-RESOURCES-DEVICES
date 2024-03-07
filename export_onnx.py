import torch.onnx
from student import *
from teacher import *
from torch import Tensor
import torch
import torch.nn as nn
import librosa
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Optional
import onnx
from menu import get_main_menu
# from data_utils import pad
from startup_config import set_random_seed
from main import W2V2_TA
import logging
import onnxruntime
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(f"Using device {device}")

args = get_main_menu()
set_random_seed(1221, args)


def perform_onnx_inference(model_path, input):
    # Convert tensor input to numpy array
    input = input.numpy()

    # Load the ONNX model
    session = onnxruntime.InferenceSession(model_path)

    # Get input information from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run the inference
    result = session.run([output_name], {input_name: input})[0]

    # Print the result (you may need to adapt this based on your model's output)
    # print("Output result:", result)
    return result


def pad(x, max_len: int = 64600) -> torch.Tensor:
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        wav_padded = pad(x).unsqueeze(0)
        output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = self.model(
            wav_padded)
        # output = self.model(wav_padded)
        # Return the probability of being spoofed
        # return output
        print("Output from the wrapper model: ", output)
        return self.softmax(output)[0][0]


input, _ = librosa.load("LA_T_1541806.wav", sr=16000)
input = torch.tensor(input).unsqueeze(0)
print(input.shape)
padded_input = pad(input).unsqueeze(0)

print("Pdaded input shape", padded_input.shape)

checkpoint = '/nfs/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_3_v10/best_checkpoint_63.pth'

model = SelfDistil_W2V2BASE_AASISTL(
    device, ssl_cpkt_path='/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt')
# model = W2V2_AASIST(device)

model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))

print("Before replace")

model.eval()

with torch.no_grad():

    before, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(
        padded_input)
    # before = model(padded_input)
    print(before)

    # Replace frontend

    model.module.ssl_model = W2V2_TA(import_fairseq_model(
        model.module.ssl_model.model
    )).to(device)

print("After replace")

model.eval()

with torch.no_grad():
    after, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, x_ssl_feat = model(
        padded_input)
    # after = model(padded_input)
    print(after)
# Export the model

save_model_path = "W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_3_v10_best63.onnx"

wrapper_model = WrapperModel(model.module).to(device)
wrapper_model.eval()

with torch.no_grad():
    out = wrapper_model(input)
    print("Output from the wrapper model")
    print(out)


torch.onnx.export(wrapper_model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  input,
                  # where to save the model (can be a file or file-like object)
                  save_model_path,
                  #   export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  #   do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  #   dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {
                  #       0: "batch_size", 1: "sequence_length"}}
                  )
# onnx_program = torch.onnx.dynamo_export(wrapper_model, padded_input)
# onnx_program.save(save_model_path)
results = perform_onnx_inference(save_model_path, input)
print(results)
torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)

# Convert results to tensor
results = torch.tensor(results)

# torch.testing.assert_close(after, results, rtol=1e-3, atol=1e-3)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
