from argparse import Namespace
from ray import tune
import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval, Dataset_ASVspoof2021_with_labels_eval
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor, Distil_W2V2BASE_AASISTL_Cosine, Distil_W2V2BASE_AASISTL, Distil_W2V2BASE_AASISTL_Regressor
from teacher import W2V2_AASIST, W2V2_AASIST_Cosine, W2V2_AASIST_Regressor
from kdtoolkit import  train_kd_cosine_loss, train_kd_mse_loss
from menu import get_hyp_tuning_menu
from utils import EarlyStopping
# from kdtoolkit import train_knowledge_distillation
from startup_config import set_random_seed
import ray
from ray.tune.schedulers import ASHAScheduler
import tempfile
from ray.train import Checkpoint

import logging

# Get the Numba logger
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)  # Set level to WARNING, ERROR, or CRITICAL

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

def train_knowledge_distillation_cosine_loss(teacher, student, train_loader,dev_loader, optimizer, hidden_rep_loss_weight, ce_loss_weight, device):
    print('Training student with knowledge distillation. hidden_rep_loss_weight: {}, ce_loss_weight: {}'.format(hidden_rep_loss_weight, ce_loss_weight))
    
    # ce_loss = nn.CrossEntropyLoss()
    
    #set objective (Loss) functions
    running_loss = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    cosine_loss = nn.CosineEmbeddingLoss()
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
            _, teacher_hidden_representation = teacher(batch_x)

            # Forward pass with the student model
        # Forward pass with the student model
        # Forward pass with the student model
        student_logits, student_hidden_representation = student(batch_x)
        

        #Soften the student logits by applying softmax first and log() second
        # Flatten the tensors from shape [30, 201, 128] to [30, 201*128]
        student_hidden_representation = student_hidden_representation.view(batch_size, -1)
        teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)

        # Now pass the reshaped tensors to cosine_loss
        hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))
        # hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_x.size(0)).to(device))
        # print("hidden_rep_loss",hidden_rep_loss)

        # Calculate the true label loss
        label_loss = criterion(student_logits, batch_y)
        # print("label_loss",label_loss)

            # Weighted sum of the two losses
        loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss
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

            batch_out,_ = student(batch_x)
           
                
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
        val_loss /= num_total

    return running_loss, val_loss

def train_knowledge_distillation_mse_loss(teacher, student, train_loader,dev_loader, optimizer, feature_map_weight, ce_loss_weight, device):
    print('Training student with knowledge distillation. hidden_rep_loss_weight: {}, ce_loss_weight: {}'.format(feature_map_weight, ce_loss_weight))
    
    # ce_loss = nn.CrossEntropyLoss()
    
    #set objective (Loss) functions
    running_loss = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    mse_loss = nn.MSELoss()
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
            _, teacher_feature_map = teacher(batch_x)
            # print("teacher_feature_map",teacher_feature_map.shape)

        # Forward pass with the student model
        student_logits, regressor_feature_map  = student(batch_x)
        # print("regressor_feature_map",regressor_feature_map.shape)
     
        #Soften the student logits by applying softmax first and log() second
        hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

        # print("hidden_rep_loss",hidden_rep_loss)

        # Calculate the true label loss
        label_loss = criterion(student_logits, batch_y)

        # print("label_loss",label_loss)

        # Weighted sum of the two losses
        loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

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

            batch_out,_ = student(batch_x)
           
                
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
        val_loss /= num_total

    return running_loss, val_loss
def train_knowledge_distillation_config(config):
    args = Namespace(database_path='/nfs/datab/hungdx/KDW2V-AASISTL/databases/', protocols_path='/nfs/datab/hungdx/KDW2V-AASISTL/protocols/', batch_size=32, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/nfs/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=False, KD_cosine=False, KD_mse=True, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)
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
        student = Distil_W2V2BASE_AASISTL(device)
        kd_method = 'KD_logits'

    elif args.KD_cosine:
        model = W2V2_AASIST_Cosine()
        student = Distil_W2V2BASE_AASISTL_Cosine(device)
        kd_method = 'KD_cosine'

    elif args.KD_mse:
        model = W2V2_AASIST_Regressor()
        student = Distil_W2V2BASE_AASISTL_Regressor(device)
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
            ## KD logits
            # running_loss, val_loss = train_knowledge_distillation(
            #     model, 
            #     student, 
            #     train_loader, 
            #     dev_loader,
            #     optimizer, 
            #     T=config["T"], 
            #     soft_target_loss_weight=config["soft_target_loss_weight"], 
            #     ce_loss_weight=config["ce_loss_weight"], 
            #     device=device
            # )

            ## KD cosine
            # running_loss, val_loss = train_knowledge_distillation_cosine_loss(
            #     model, 
            #     student, 
            #     train_loader, 
            #     dev_loader,
            #     optimizer, 
            #     hidden_rep_loss_weight=config["hidden_rep_loss_weight"], 
            #     ce_loss_weight=config["ce_loss_weight"], 
            #     device=device
            # )

            ## KD mse
            running_loss, val_loss = train_knowledge_distillation_mse_loss(
                model,
                student,
                train_loader,
                dev_loader,
                optimizer,
                feature_map_weight=config["feature_map_weight"],
                ce_loss_weight=config["ce_loss_weight"],
                device=device
            )

            # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #     path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            #     torch.save(
            #         (student.state_dict(), optimizer.state_dict()), path
            #     )
            #     checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            #     ray.train.report(
            #         {'train_loss': running_loss, 'val_loss': val_loss},
            #         checkpoint=checkpoint,
            #     )
            # # Report both training and validation loss to Tune
            metrics = {'train_loss': running_loss, 'val_loss': val_loss}
            # Report both training and validation loss to Tune
            ray.train.report(metrics)


def test_best_model(best_result, model_type="KD_base_cosine"):
    
    args = Namespace(database_path='/nfs/datab/hungdx/KDW2V-AASISTL/databases/', protocols_path='/nfs/datab/hungdx/KDW2V-AASISTL/protocols/', batch_size=32, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/nfs/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=False, KD_cosine=False, KD_mse=True, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)
    track = 'DF'
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_type == "KD_base_cosine":
        best_trained_model = Distil_W2V2BASE_AASISTL_Cosine(device)
    else:
        raise ValueError("Invalid model type")
    
    num_eval_samples = 150000
    best_trained_model.to(device)
    best_trained_model = nn.DataParallel(best_trained_model)
    
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    best_trained_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    labels, file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True, num_eval_samples=num_eval_samples)
    print('no. of eval trials',len(file_eval))
    eval_set=Dataset_ASVspoof2021_with_labels_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(track)) ,labels = labels)

    correct = 0
    total = 0
    batch_size = 200
    testloader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.inference_mode():
        for batch_x,utt_id, batch_labels in testloader:
            batch_out, _ = best_trained_model(batch_x)
            
            # Using softmax to get the probability of the positive class
            batch_out = nn.Softmax(dim=1)(batch_out)[:,0]
            
            # If the probability is greater than 0.5, then the model predicts the sample is fake or 0 (spoof)
            # Let's compare the predictions with the ground truth labels
            predicted = (batch_out > 0.5).type(torch.int64)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            

    print("Best trial test set accuracy: {}".format(correct / total))
# Define the search space
search_space = {

    ## =================== KD loss ===================
    # "T": tune.loguniform(1.0, 5.0),
    # "soft_target_loss_weight": tune.uniform(0.0, 1.0),
    # "ce_loss_weight": tune.uniform(0.0, 1.0),
    ## =================== KD cosine loss ===================
    # "hidden_rep_loss_weight": tune.uniform(0.0, 1.0),
    # "ce_loss_weight": tune.uniform(0.0, 1.0),
    ## =================== KD mse loss ===================
    "feature_map_weight": tune.uniform(0.0, 1.0),
    "ce_loss_weight": tune.uniform(0.0, 1.0),
}

# Run the hyperparameter search
analysis = tune.run(
    train_knowledge_distillation_config, 
    config=search_space, 
    num_samples=40,  # Number of samples to run
    resources_per_trial={"cpu": 2, "gpu": 1},  # Resources to allocate per trial
)

# Get the best configuration
best_config = analysis.get_best_config(metric="loss", mode="min")
print(best_config)

