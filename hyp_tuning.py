from argparse import Namespace
from ray import tune
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor
from teacher import W2V2_AASIST, W2V2_AASIST_Cosine, W2V2_AASIST_Regressor
from kdtoolkit import  train_kd_cosine_loss, train_kd_mse_loss
from menu import get_hyp_tuning_menu
from utils import EarlyStopping
# from kdtoolkit import train_knowledge_distillation
from startup_config import set_random_seed
import ray

MAX_EPOCHS = 5

def train_knowledge_distillation(teacher, student, train_loader,dev_loader, optimizer, T, soft_target_loss_weight, ce_loss_weight, device):
    print('Training student with knowledge distillation. T: {}, soft_target_loss_weight: {}, ce_loss_weight: {}'.format(T, soft_target_loss_weight, ce_loss_weight))
    
    # ce_loss = nn.CrossEntropyLoss()
    
    #set objective (Loss) functions
    running_loss = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    num_total = 0.0

    val_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size
        optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            teacher_logits = teacher(batch_x)

            # Forward pass with the student model
        student_logits = student(batch_x)

            #Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)

            # print("soft_targets_loss",soft_targets_loss)

            # Calculate the true label loss
        label_loss = criterion(student_logits, batch_y)
            # print("label_loss",label_loss)

                # Weighted sum of the two losses
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
            # print("loss",loss)

        loss.backward()
        optimizer.step()

        running_loss += (loss.item() * batch_size)
    running_loss /= num_total

    val_loss = 0.0
    num_total = 0.0
    student.eval()

    print('Validating student...')
    with torch.inference_mode():
        for batch_x, batch_y in dev_loader:
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = student(batch_x)
           
                
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
        val_loss /= num_total

    return running_loss, val_loss

def train_knowledge_distillation_config(config):
    args = Namespace(database_path='/nfs/datab/hungdx/KDW2V-AASISTL/databases/', protocols_path='/nfs/datab/hungdx/KDW2V-AASISTL/protocols/', batch_size=32, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/nfs/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=True, KD_cosine=False, KD_mse=False, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)
    set_random_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn

        # define dev (validation) dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)

    del dev_set,d_label_dev



    if args.KD_logits:
        model = W2V2_AASIST()
        student = Distil_W2V2_AASISTL(device)
        kd_method = 'KD_logits'

    elif args.KD_cosine:
        model = W2V2_AASIST_Cosine()
        student = Distil_W2V2_AASISTL_Cosine(device)
        kd_method = 'KD_cosine'

    elif args.KD_mse:
        model = W2V2_AASIST_Regressor()
        student = Distil_W2V2_AASISTL_Regressor(device)
        kd_method = 'KD_mse'

    else:
        raise ValueError('Invalid KD method given')

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =nn.DataParallel(model).to(device)
    print('Teacher nb_params:',nb_params)
        
    nb_params = sum([param.view(-1).size()[0] for param in student.parameters()])
    student = nn.DataParallel(student).to(device)
    print('Student nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

        for epoch in range(MAX_EPOCHS):
            running_loss, val_loss = train_knowledge_distillation(
                model, 
                student, 
                train_loader, 
                dev_loader,
                optimizer, 
                T=config["T"], 
                soft_target_loss_weight=config["soft_target_loss_weight"], 
                ce_loss_weight=config["ce_loss_weight"], 
                device=device
            )

            # Debugging
            # train_knowledge_distillation(model, 
            #     student, 
            #     train_loader, 
            #     optimizer, 1, 2, 3, device=device)

            # Report both training and validation loss to Tune
            metrics = {'train_loss': running_loss, 'val_loss': val_loss}
            # Report both training and validation loss to Tune
            ray.train.report(metrics)

# Define the search space
search_space = {
    "T": tune.loguniform(1.0, 5.0),
    "soft_target_loss_weight": tune.uniform(0.0, 1.0),
    "ce_loss_weight": tune.uniform(0.0, 1.0),
}

# Run the hyperparameter search
analysis = tune.run(
    train_knowledge_distillation_config, 
    config=search_space, 
    num_samples=20,  # Number of samples to run
    resources_per_trial={"cpu": 2, "gpu": 1},  # Resources to allocate per trial
)

# Get the best configuration
best_config = analysis.get_best_config(metric="loss", mode="min")
print(best_config)
