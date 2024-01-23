import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torch import nn
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor
import numpy as np
from torch import Tensor
import onnx
import torch.onnx
import onnxruntime
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"

model_path="Distil_W2V2_AASISTL_Regressor3.onnx"

# onnx_model = onnx.load(model_path)
# print("load the model")
# onnx.checker.check_model(onnx_model)

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