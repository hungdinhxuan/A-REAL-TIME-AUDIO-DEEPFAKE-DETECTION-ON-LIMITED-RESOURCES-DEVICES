from student import *
from teacher import *
from data_utils import *
from torch import Tensor
import torch
import torch.nn as nn
import librosa
import os
import sys
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Optional
from menu import get_main_menu
#import onnxruntime
# from data_utils import pad
from startup_config import set_random_seed
from main import W2V2_TA
import logging
from torchinfo import summary
import torch.onnx
import yaml
from torchdistill.models.registry import get_model
from wav2vec2_vib import Model as W2V2_VIB

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(f"Using device {device}")

args = get_main_menu()

PADDING_SIZE = int(args.padding * 16000)  # 1s = 16000 samples
print("PADDING", PADDING_SIZE)
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


def pad2(x, max_len: int = PADDING_SIZE):
    """
    Trace-friendly padding function optimized for serving/inference.
    Handles both truncation and repetitive padding without TracerWarnings.
    """
    # Handle both 1D and 2D tensors consistently
    if x.dim() == 2:
        x = x.squeeze(0)
    
    x_len = x.size(0)
    
    # Method 1: Use torch.tile (PyTorch >= 1.7) - most efficient and trace-friendly
    # Calculate a safe number of repetitions
    # For typical audio (16kHz, 4sec = 64k samples), even with 100ms input (1.6k samples)
    # we'd need ~40 repetitions. Use 50 for safety margin.
    num_reps = max(1, (max_len // 1000) + 10)  # Conservative estimate
    
    # torch.tile is the most efficient way to repeat tensors
    tiled = torch.tile(x, (num_reps,))
    
    # Always truncate to exact length
    return tiled[:max_len]

def pad(x, max_len: int = PADDING_SIZE):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x

@torch.jit.script
def make_sure_2d(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x

# def pad2(x, max_len: int = PADDING_SIZE):
#     '''
#     Maximum length of the input is 10000
#     '''
#     pass

class WrapperScaledModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.threshold = -4.35825252532959

        self.min_score = -5.1182541847229

        self.max_score = 4.072232246398926

        print('WrapperScaledModel: ', self.threshold,
              self.min_score, self.max_score)

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


class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        print('WrapperModel')

    def forward(self, x):
        wav_padded = pad(x).unsqueeze(0)
        output = self.model(wav_padded)
        return self.softmax(output)[0][0]

        
class WrapperModel_NOPAD(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        print('WrapperModel_NOPAD')

    def forward(self, x):
        #wav_padded = pad(x).unsqueeze(0)
        x = make_sure_2d(x)
        with torch.no_grad():
            output = self.model(x)
        return self.softmax(output)[0][0]



# Load spoofed sample
input, _ = librosa.load(
    "/datad/Datasets/moreko/wavs/09MKIS0040_12815.wav", sr=16000)  # Bona fide
input = torch.tensor(input).unsqueeze(0)
# input = torch.zeros(1, 64600)

print(input.shape)
padded_input = pad(input).unsqueeze(0) if PADDING_SIZE > 0 else input

checkpoint = args.student_model_path

# Init Linear model
# model = Distil_W2V2BASE_Linear(
#     device, ssl_cpkt_path="/datad/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

# Init VIB model
# model = Distil_W2V2BASE_VIB(
#     device, ssl_cpkt_path="/datad/hungdx/KDW2V-AASISTL/wav2vec_small.pt")


# Latest model
with open(args.yaml, 'r') as f:
    config = yaml.safe_load(f)
student_model_name = config['model']['student']['name']

model = get_model(
    student_model_name, device=device, **config['model']['student']['kwargs']).to(device)

model = nn.DataParallel(model).to(device)
# model = W2V2_VIB(device, ssl_cpkt_path='/datad/hungdx/KDW2V-AASISTL/pretrained/xlsr2_300m.pt')


# Load checkpoint
model.load_state_dict(torch.load(
    checkpoint, map_location=device))

print("Loaded model from ", checkpoint)

model.eval()

print("Before replace")
with torch.no_grad():
    before = model(padded_input)
    print(before)

model.module.front_end = W2V2_TA(import_fairseq_model(
    model.module.front_end.model
)).to(device)
# model.ssl_model = W2V2_TA(import_fairseq_model(
#     model.ssl_model.model
# )).to(device)

model.eval()
print("After replace")
with torch.no_grad():
    after = model(padded_input)
    print(after)

# ================================================================ Testing fusion model ================================================================
# model2 = Distil_W2V2BASE_Linear(
#     device, ssl_cpkt_path='/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt')
# model2 = nn.DataParallel(model2).to(device)
# model2.load_state_dict(torch.load(
#     "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_Linear_DKDLoss_cnsl_noaudiomentations/best_checkpoint_39.pth", map_location=device))

# model2.module.ssl_model = W2V2_TA(import_fairseq_model(
#     model2.module.ssl_model.model
# )).to(device)

# model2.eval()

# # Fusion model
# fusion_model = WrapperFusionModel(nn.ModuleList([model, model2])).to(device)
# fusion_model.eval()
# with torch.no_grad():
#     print("After fusion")
#     after = fusion_model(padded_input)
#     print(after)

# # Summary of the model
# summary(fusion_model, input_size=(1, 64600))
# ================================================================ Testing fusion model ================================================================

# Scriptable

# # Wrapper model

if args.scale_export:
    print("Using scaled model")
    model_fp32 = WrapperScaledModel(model.module).to(device)
elif PADDING_SIZE < 0:
    print("Using no pad model")
    model_fp32 = WrapperModel_NOPAD(model.module).to(device)
else:
    print("Using normal model")
    model_fp32 = WrapperModel(model.module).to(device)
    # model_fp32 = WrapperModel(model).to(device)

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
 
if args.qat:
    print("Quantizing model")
    comment += "_qat"
    #torch.backends.quantized.engine = 'qnnpack'

    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8)
    
    padded_input = padded_input.squeeze(0)
    print("Padded input shape", padded_input.shape)
    scripted_model = torch.jit.trace(quantized_model, padded_input)
    # Save scripted model
    scripted_model_path = SAVE_MODEL_PATH % "qat_scripted"
    torch.jit.save(scripted_model, scripted_model_path)
    print("Saved scripted model to ", scripted_model_path)
    
    from torch._C import _MobileOptimizerType as MobileOptimizerType
    
    # 4. Optimize for mobile
    # optimized_model = optimize_for_mobile(scripted_model, optimization_blocklist={
    #         #MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
    #         MobileOptimizerType.INSERT_FOLD_PREPACK_OPS
    # })
    optimized_model = optimize_for_mobile(scripted_model)
    
    # Save quantized model
    quantized_model_path = SAVE_MODEL_PATH % "qat_mobile"
    torch.jit.save(optimized_model, quantized_model_path)
    print("Saved quantized model to ", quantized_model_path)
    
    # Save lite model
    lite_model_path = quantized_model_path.split(".pt")[0] + "_lite.ptl"
    optimized_model._save_for_lite_interpreter(lite_model_path)
    print("Saved lite model to ", lite_model_path)
    
    # Testing lite model
    
    
    
   
    print("Done quantizing")
    import sys
    sys.exit(0)

if args.executorch:
    print("Exporting executorch model")
    comment += "_executorch"
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from torch.export import Dim, export
    dynamic_shapes = {
        "x": Dim("length", min=16000, max=160000)
    }
    inputs = (torch.randn(1, 64000),)
    exported_program = export(model_fp32, inputs, dynamic_shapes=dynamic_shapes)
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    ).to_executorch()
    print("Executorch program")
    with open("model.pte", "wb") as f:
        f.write(executorch_program.buffer)
        
    # Testing executorch program
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program("model.pte")
    method = program.load_method("forward")
    print("Executorch program output")
    outputs = method.execute([inputs])
    print(outputs)
    sys.exit(0)
 


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


print("Before trace")
padded_input = padded_input.squeeze(0)
with torch.no_grad():
    before = model_fp32(padded_input)
    print(before)

# Script
#jit_model = torch.jit.script(model_fp32)
# Trace


print("Padded input shape", padded_input.shape)

jit_model = torch.jit.trace(model_fp32, padded_input)

with torch.no_grad():
    jit_out = jit_model(padded_input)
    print("JIT model output")
    print(jit_out)

SAVE_MODEL_PATH_LAPTOP = SAVE_MODEL_PATH % "jit"
# optimized_jit_for_inference = torch.jit.optimize_for_inference(jit_model)
# torch.jit.save(jit_model, SAVE_MODEL_PATH_LAPTOP)
# print("Saved model to ", SAVE_MODEL_PATH_LAPTOP)

from torch._C import _MobileOptimizerType as MobileOptimizerType
opt_model = optimize_for_mobile(jit_model, optimization_blocklist={
            #MobileOptimizerType.HOIST_CONV_PACKED_PARAMS,
            MobileOptimizerType.INSERT_FOLD_PREPACK_OPS
    })
print("After optimize_for_mobile")
with torch.no_grad():
    after = opt_model(padded_input)
    print(after)

# Save optimized model
SAVE_MODEL_PATH_MOBILE = SAVE_MODEL_PATH % "mobile"
opt_model.save(SAVE_MODEL_PATH_MOBILE)
print("Saved model to ", SAVE_MODEL_PATH_MOBILE)


# Save
opt_model._save_for_lite_interpreter(SAVE_MODEL_PATH_MOBILE.split(".pt")[0] + "_lite.ptl")
print("Done~")
