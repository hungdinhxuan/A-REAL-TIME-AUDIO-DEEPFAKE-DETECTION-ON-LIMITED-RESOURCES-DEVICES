from student import Distil_W2V2BASEHG_AASISTL_Self_KD, Distil_W2V2BASE_AASISTL_Self_KD_Teacher
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
# Quantization code
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from torch.utils.data import DataLoader
from neural_compressor.utils.pytorch import load
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
from menu import get_main_menu

args = get_main_menu()


model = Distil_W2V2BASE_AASISTL_Self_KD_Teacher(device)
model = torch.nn.DataParallel(model).to(device)

model_path = "/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_40_1e-06_self_KD_teacher_W2VBaseHG/best_checkpoint_42.pth"

# Load the model
model.load_state_dict(torch.load(model_path,map_location=device))
print("Loaded model from {}".format(model_path))
model.eval()

print(model)

# @torch.jit.script
# def pad(x, max_len: int = 64600) -> Tensor:
#     x_len = torch.tensor(x.shape[0])
#     max_len = torch.tensor(max_len)

#     if torch.ge(x_len, max_len).item():
#         return x[:max_len]
#         # need to pad
#     num_repeats = int((max_len / x_len).ceil().item())
        
#     padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
#     return padded_x
# # Define a wrapper model
# class WrapperModel(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.softmax = nn.Softmax(dim=1)

    
#     def forward(self, x):
#         wav_padded = pad(x).unsqueeze(0) if x.dim() == 1 else x
#         output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = self.model(wav_padded)
#         return output

# # Inference test
# input = torch.randn(1, 16000).to(device)

# model = WrapperModel(model.module).to(device)
# # model.eval()

# #Dynamic quantization work well

# quantized_model = torch.quantization.quantize_dynamic(
#     model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

# # Save the quantized model


# scripted_model = torch.jit.script(quantized_model)

# # Save the quantized model
# # scripted_model.save("Distil_W2V2BASEHG_AASISTL_Self_KD_quantized.pt")

# # def print_tensors(model):
# #     for name, param in model.named_parameters():
# #         print(name, param.data)

# # print_tensors(scripted_model)

# # torch.jit.save(scripted_model, "Distil_W2V2BASEHG_AASISTL_Self_KD_quantized.pt")

# optimized_model = optimize_for_mobile(scripted_model)

# optimized_model.save("Distil_W2V2BASEHG_AASISTL_Self_KD_quantized_optimized.pt")
# print('Done saved')
# # print(Tensor(data))
# res = optimized_model(input)
# print(res)
# # print('Result:', optimized_model(Tensor(data)))
# # optimized_model._save_for_lite_interpreter("Distil_W2V2BASEHG_AASISTL_Self_KD_quantized_optimized.ptl")
# # prefix_2021 = 'ASVspoof2021.{}'.format(args.track)
# # _,file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(args.track,prefix_2021)),is_train=False,is_eval=True, num_eval_samples=args.num_eval_samples)
# # print('no. of eval trials',len(file_eval))
# # eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
# # eval_loader = DataLoader(eval_set, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

# # q_model = quantization.fit(
# #     model=model,
# #     conf=PostTrainingQuantConfig(approach="auto"),
# #     calib_dataloader=eval_loader,
# # )
# # q_model.save("./Distil_W2V2BASEHG_AASISTL_Self_KD_auto_quantized")

# # Load quantized model

# # conf = PostTrainingQuantConfig(
# #     approach="weight_only",
# #     op_type_dict={
# #         ".*": {  # re.match
# #             "weight": {
# #                 "bits": 4,  # 1-8 bit
# #                 "group_size": -1,  # -1 (per-channel)
# #                 "scheme": "sym",
# #                 "algorithm": "RTN",
# #                 "dtype": "nf4",
# #             },
            
# #         },
# #     },
# #     recipes={
# #         'rtn_args':{'enable_full_range': True, 'enable_mse_search': True},
# #         # 'gptq_args':{'percdamp': 0.01, 'actorder':True, 'block_size': 128, 'nsamples': 128, 'use_full_length': False},
# #         # 'awq_args':{'enable_auto_scale': True, 'enable_mse_search': True, 'n_blocks': 5},
# #     },
# # )
# # q_model = quantization.fit(model, conf)
# # q_model.save("saved_results")
# # compressed_model = q_model.export_compressed_model()
# # torch.save(compressed_model.state_dict(), "compressed_model.pt")

# # model = load("./Distil_W2V2BASEHG_AASISTL_Self_KD_auto_quantized", model)
# # print("Loaded quantized model")
# # model.eval()

# # # Test inference
# # print("Inference test")

# # with torch.no_grad():
# #     output = model(torch.randn(1, 16000).to(device))
# #     print("output: ", output)
# #     print("Done! Test OK!")

# # Script model and save
# # scripted_model = torch.jit.script(model)
# # optimized_model = optimize_for_mobile(scripted_model)
# # optimized_model._save_for_lite_interpreter("./Distil_W2V2BASEHG_AASISTL_Self_KD_auto_quantized/best.ptl")