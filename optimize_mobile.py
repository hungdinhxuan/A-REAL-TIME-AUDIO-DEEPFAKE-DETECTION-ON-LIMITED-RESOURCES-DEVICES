from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = Distil_W2V2_AASISTL_Regressor(device="cpu")
model = nn.DataParallel(model).to("cpu")
# Load quantized model
model.load_state_dict(torch.load("/nfs/datab/hungdx/KDW2V-AASISTL/compressed_model.pt",map_location="cpu"), strict=False)
with torch.inference_mode():
    result, _ = model(torch.randn(1,64000))
    print(result)

# trace_model = torch.jit.trace(model, torch.randn(1,64000))
# optimized_trace_model = optimize_for_mobile(trace_model)

# # Save the optimized model
# optimized_trace_model._save_for_lite_interpreter("/nfs/datab/hungdx/KDW2V-AASISTL/KD_mse_auto_quantized_model_ckp23/best_model.ptl")