import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval, Dataset_ASVspoof2019_train_emb
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor, Distil_W2V2BASE_AASISTL, Distil_W2V2BASE_AASISTL_Cosine, Distil_W2V2BASE_AASISTL_Regressor, Distil_W2V2FTBASE_AASISTL, Distil_W2V2BASE_AASISTL_Self_KD, Distil_W2V2BASE_AASISTL_Self_KD_Teacher, Distil_W2V2BASEHG_AASISTL_Self_KD, Distil_W2V2BASEHG_AASISTL_Self_KD_Teacher, Distil_W2V2BASEHG_AASISTL_Self_KD_Teacher_Drop, Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher
from teacher import W2V2_AASIST, W2V2_AASIST_Cosine, W2V2_AASIST_Regressor
from kdtoolkit import train_knowledge_distillation, train_kd_cosine_loss, train_kd_mse_loss, kd_loss_function, feature_loss_function
from menu import get_main_menu
from utils import EarlyStopping, AverageMeter
from torch.optim.lr_scheduler import StepLR
from main import get_train_dev_dataloader
device = "cuda" if torch.cuda.is_available() else "cpu"
folder_to_extract = "./teacher_cosine_emb"
os.makedirs(folder_to_extract, exist_ok=True)
args = get_main_menu()

d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
    
print('no. of training trials',len(file_train))
    
train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
    
train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=2, shuffle=True,drop_last = True, pin_memory=True, pin_memory_device=device)
    
del train_set,d_label_trn


# Define a new model
teacher_model = W2V2_AASIST_Cosine(device)
teacher_model = torch.nn.DataParallel(teacher_model).to(device)
teacher_model.load_state_dict(torch.load(args.model_path))
teacher_model.eval()
with torch.no_grad():
    for batch_x,batch_y, batch_utt in train_loader:
        _, teacher_hidden_representation = teacher_model(batch_x)
        # Save teacher hidden representation
        for i, utt_id in enumerate(batch_utt):
            save_path = os.path.join(folder_to_extract, utt_id+".pt")
            if os.path.exists(save_path):
                print("Warning: {} already exists".format(save_path))
            else:
                torch.save(teacher_hidden_representation[i], save_path)
                print("Save teacher hidden representation to {}".format(save_path))
      

