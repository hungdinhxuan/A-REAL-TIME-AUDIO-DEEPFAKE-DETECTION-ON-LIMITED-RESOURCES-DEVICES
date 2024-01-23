import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
import numpy as np
from torch import Tensor
import onnx
import torch.onnx
import onnxruntime
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"

# onnx_model = onnx.load("Distil_W2V2_AASISTL_Regressor2.onnx")
# print("load the model")
# onnx.checker.check_model(onnx_model)


model_path="static_quant3.onnx"

def generate_random_data():
    # Generate random input data
    return np.random.rand(1,64000).astype(np.float32)

def perform_onnx_inference(model_path):
    # Load the ONNX model
    session = onnxruntime.InferenceSession(model_path)

    # Get input information from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Generate random input data based on the input shape
    data = generate_random_data()

    # Run the inference
    result = session.run([output_name], {input_name: data})[0]

    # Print the result (you may need to adapt this based on your model's output)
    #print("Output result:", result)
    return result
result=perform_onnx_inference(model_path)
print(result)



