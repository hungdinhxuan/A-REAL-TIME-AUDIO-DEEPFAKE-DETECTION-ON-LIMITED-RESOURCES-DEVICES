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
from torch.utils.mobile_optimizer import optimize_for_mobile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"

model = Distil_W2V2_AASISTL_Regressor(device=device)
model = nn.DataParallel(model).to(device)

model_path = "/nfs/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_30_1e-06_KD_mse/best_checkpoint_47.pth"

# Load the model
model.load_state_dict(torch.load(model_path,map_location=device))
print("Loaded model from {}".format(model_path))

#Define a wrapper model
class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def pad(self, x, max_len: int = 64600) -> Tensor:
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
        return padded_x
    
    def forward(self, x: Tensor):
        wav_padded = self.pad(x).unsqueeze(0)
        output, regressor_output = self.model(wav_padded)

        # Softmax the output
        output = nn.functional.softmax(output, dim=1)
        print("output",output)

        # Final result in % fake
        return output[0][0]

#Inference
input = torch.randn(1, 64000)
model = WrapperModel(model).to(device)

scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter("Distil_W2V2_AASISTL_Regressor.ptl")
print("Done _save_for_lite_interpreter")

with torch.inference_mode():
    print(optimized_model(input))

# model = model.module if isinstance(model, nn.DataParallel) else model

# print("converting to onnx model...")

# # convert to ONNX
# onnx_path = "Distil_W2V2_AASISTL_Regressor3.onnx"
# model=torch.onnx.export(model, input, onnx_path, export_params=True)

# print(f"Model exported to {onnx_path}")

#with torch.inference_mode():
    #print(model(input))


            # Create dummy input data
            # for inputs, targets in train_loader:
            #     print(inputs.shape)
            #     break
            # inputs = torch.randn(64, 64600)  # Assuming input size is (3, 32, 32)

            # # Create dummy target data
            # targets = torch.randint(0, 2, (64,))  # Assuming binary classification

            # # Create a TensorDataset
            # dataset = TensorDataset(inputs, targets)

            # # Create a DataLoader
            # train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)