from student import *
from teacher import *
from torch import Tensor
import torch
import torch.nn as nn
import librosa
import os
import sys
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Optional
from menu import get_main_menu
import onnxruntime
# from data_utils import pad
from startup_config import set_random_seed
from main import W2V2_TA
import logging
import torch.onnx
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PADDING_SIZE = 64600


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


def pad(x, max_len: int = PADDING_SIZE):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(f"Using device {device}")

args = get_main_menu()
set_random_seed(1221, args)


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.threshold = -1.3279
        self.min_score = -2.36
        self.max_score = 5.69

    def scaled_likelihood(self, score: float) -> float:
        '''
        if score =< threshold, then scale it to [0, 50)
        if score > threshold, then scale it to [50, 100]
        based on the min_score and max_score
        '''
        if score <= self.threshold:
            scaled = (score - self.min_score) / \
                (self.threshold - self.min_score) * 50.0
        else:
            scaled = 50.0 + (score - self.threshold) / \
                (self.max_score - self.threshold) * 50.0

        if scaled < 0.0:
            scaled = 0.0
        elif scaled > 100.0:
            scaled = 100.0
        return scaled

    def forward(self, x) -> Tensor:

        wav_padded = pad(x).unsqueeze(0)
        output = self.model(wav_padded)
        print(output)
        # Return the probability of being spoofed
        spoof_score = (
            100.0 - self.scaled_likelihood(output[0][1].item())) / 100.0

        # Convert to tensor
        spoof_score = torch.tensor(spoof_score)
        return spoof_score


# Load spoofed sample
input, _ = librosa.load("LA_T_1541806.wav", sr=16000)
input = torch.tensor(input).unsqueeze(0)
# input = torch.zeros(1, 64600)

print(input.shape)
padded_input = pad(input).unsqueeze(0)

checkpoint = args.student_model_path

# Load another model
model = Distil_W2V2BASE_AASISTL(
    device, ssl_cpkt_path='/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt')

model = nn.DataParallel(model).to(device)

model.load_state_dict(torch.load(
    checkpoint, map_location=device), strict=False)

model.module.ssl_model = W2V2_TA(import_fairseq_model(
    model.module.ssl_model.model
)).to(device)

model.eval()

print("After replace")
with torch.no_grad():
    after = model(padded_input)
    print(after)

# Scriptable
model_fp32 = WrapperModel(model.module).to(device)
model_fp32.eval()

# Get last file name from path
# Get the second last file name from path
second_last = os.path.basename(os.path.dirname(checkpoint))
print("Second last", second_last)

comment = args.comment
if comment is None:
    comment = ""

SAVE_MODEL_PATH = f"{second_last}_{os.path.basename(checkpoint).split('.')[0]}_{comment}%s.pt"
os.makedirs("./exports", exist_ok=True)
SAVE_MODEL_PATH = os.path.join("./exports", SAVE_MODEL_PATH)

if args.bf16:
    SAVE_MODEL_PATH_LAPTOP_BF16 = SAVE_MODEL_PATH % "bf16"
    with torch.cpu.amp.autocast():
        model_bf16 = torch.jit.script(model_fp32)
        model_bf16 = torch.jit.freeze(model_bf16)

        with torch.no_grad():
            y = model_bf16(padded_input)
            print("After bf16 autotrace")
            print(y)

        optimized_for_inference = torch.jit.optimize_for_inference(model_bf16)
        torch.jit.save(
            optimized_for_inference, SAVE_MODEL_PATH_LAPTOP_BF16)
        print("Saved model to ", SAVE_MODEL_PATH_LAPTOP_BF16)

    # Save optimized model for mobile
    SAVE_MODEL_PATH_MOBILE_BF16 = SAVE_MODEL_PATH % "mobile_bf16"
    opt_model = optimize_for_mobile(optimized_for_inference)
    # print("After optimize_for_mobile")
    # with torch.no_grad():
    #     after = opt_model(input)
    #     print(after)
    opt_model.save(SAVE_MODEL_PATH_MOBILE_BF16)
    print("Saving optimized model to ", SAVE_MODEL_PATH_MOBILE_BF16)


if args.onnx:

    SAVE_MODEL_ONNX_PATH = SAVE_MODEL_PATH.replace(".pt", ".onnx")
    model_fp32_jit = torch.jit.script(model_fp32)
    model_fp32_jit = torch.jit.freeze(model_fp32_jit)

    torch.onnx.export(model_fp32_jit,               # model being run
                      # model input (or a tuple for multiple inputs)
                      input,
                      # where to save the model (can be a file or file-like object)
                      SAVE_MODEL_ONNX_PATH,
                      #   export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=14,          # the ONNX version to export the model to
                      #   do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      #   dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {
                      #       0: "batch_size", 1: "sequence_length"}}
                      )
    results = perform_onnx_inference(SAVE_MODEL_ONNX_PATH, input)
    print("ONNX results")
    print(results)

jit_model = torch.jit.script(model_fp32)

with torch.no_grad():
    jit_out = jit_model(input)
    print("JIT model output")
    print(jit_out)

SAVE_MODEL_PATH_LAPTOP = SAVE_MODEL_PATH % "jit"
# optimized_jit_for_inference = torch.jit.optimize_for_inference(jit_model)
torch.jit.save(jit_model, SAVE_MODEL_PATH_LAPTOP)
print("Saved model to ", SAVE_MODEL_PATH_LAPTOP)


# opt_model = optimize_for_mobile(jit_model)
# print("After optimize_for_mobile")
# with torch.no_grad():
#     after = opt_model(input)
#     print(after)

# # Save optimized model
# SAVE_MODEL_PATH_MOBILE = SAVE_MODEL_PATH % "mobile"
# opt_model.save(SAVE_MODEL_PATH_MOBILE)
# print("Saved model to ", SAVE_MODEL_PATH_MOBILE)


# # Save
# # opt_model._save_for_lite_interpreter("W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_best_checkpoint_126.ptl")
# print("Done~")
