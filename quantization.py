import sys
import os
import torch
from torch import nn
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor
from teacher import W2V2_AASIST, W2V2_AASIST_Cosine, W2V2_AASIST_Regressor
from kdtoolkit import train_knowledge_distillation, train_kd_cosine_loss, train_kd_mse_loss
from menu import get_model
from utils import EarlyStopping
from neural_compressor import PostTrainingQuantConfig, quantization
from torch.utils.data import DataLoader
from neural_compressor.config import PostTrainingQuantConfig
from torch.utils.mobile_optimizer import optimize_for_mobile
os.environ["CUDA_VISIBLE_DEVICES"] = ""

args = get_model()

device = "cpu"

model = Distil_W2V2_AASISTL_Regressor(device=device)
model = nn.DataParallel(model).to(device)

model.load_state_dict(torch.load("/nfs/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_30_1e-06_KD_mse/best_checkpoint_23.pth",map_location=device))

# Calibrate dataset
d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
print('no. of training trials',len(file_train))
train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=False)


conf =  PostTrainingQuantConfig(
    precision="fp8_e5m2",
    calibration_sampling_size=[300],
    batchnorm_calibration_sampling_size=[3000],
)
q_model = quantization.fit(model,
                               conf=conf,
                               calib_dataloader=train_loader)
# q_model.save("./KD_mse_auto_quantized_model_ckp23")

# Optimize for mobile qmodel
print("Optimizing for mobile")

compressed_model = q_model.export_compressed_model()
torch.save(compressed_model.state_dict(), "compressed_model.pt")



# trace_model = torch.jit.trace(q_model, torch.randn(1,64000))
# optimized_trace_model = optimize_for_mobile(trace_model)
# os.makedirs("./KD_mse_weight_only_quantized_model_ckp23", exist_ok=True)
# optimized_trace_model._save_for_lite_interpreter("./KD_mse_weight_only_quantized_model_ckp23/best_model.ptl")

# model.load_state_dict(torch.load("/nfs/datab/hungdx/KDW2V-AASISTL/output/best_model.pt",map_location=device), strict=False)

# print(model(torch.randn(1,64000)))

