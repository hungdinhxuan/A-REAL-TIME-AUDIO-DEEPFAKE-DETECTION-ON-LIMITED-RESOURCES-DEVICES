import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import onnx
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.quantization.calibrate import CalibrationDataReader
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"

model_path="Regressor3.onnx"
model_quant = 'static_quant3.onnx'

# dummy data
maxN = 100
imgs = [np.random.rand(1,64000).astype(np.float32)
        for i in range(maxN)]

class DataReader(CalibrationDataReader):
    def __init__(self, input_name, imgs):
        self.input_name = input_name
        self.data = imgs
        self.pos = -1

    def get_next(self):
        if self.pos >= len(self.data) - 1:
            return None
        self.pos += 1
        return {self.input_name: self.data[self.pos]}

    def rewind(self):
        self.pos = -1

input_name="x.1"
quantize_static(model_path,model_quant,calibration_data_reader=DataReader(input_name, imgs))
