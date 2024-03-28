from argparse import Namespace
from torchdistill.models.registry import get_model
from torchdistill.losses.registry import get_mid_level_loss
from torchdistill.core.forward_hook import ForwardHookManager
from ray import tune
import sys
import os
import torch
from data_utils import *
from student import *
from teacher import *
from torch import nn
from torch.utils.data import DataLoader
from startup_config import set_random_seed
from kdtoolkit import  train_kd_cosine_loss, train_kd_mse_loss
from menu import get_hyp_tuning_menu
from utils import EarlyStopping
# from kdtoolkit import train_knowledge_distillation
from startup_config import set_random_seed
import ray
from ray.tune.schedulers import ASHAScheduler
import tempfile
from ray.train import Checkpoint
from main import self_KD_train_epoch, self_KD_val_epoch
import logging
import yaml
from main import get_train_dev_dataloader
from tqdm import tqdm
from kdtoolkit import kd_loss_function, feature_loss_function
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.train import RunConfig
from ray.tune import Stopper
from ray import train, tune
# Get the Numba logger 
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)  # Set level to WARNING, ERROR, or CRITICAL

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ['WORLD_SIZE'] = '4'
# os.environ['RANK'] = '0'

# print("hello1")


# dist.init_process_group(backend='nccl')
# print("hello")
MAX_EPOCHS = 50

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural studentwork"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

def self_KD_teacher_val_epoch(dev_loader, model, device):
    logger.info('Validation Teacher self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    num_correct = 0.0

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)
            graph_loss_S = criterion(graph_output_S, batch_y)
            graph_loss_T = criterion(graph_output_T, batch_y)
            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T
            
            total_loss = total_label_loss

        
            val_loss += (total_loss.item() * batch_size)

            probabilities = F.softmax(batch_out, dim=1)
            predicted_labels = (probabilities[:,0] >= 0.5).int()
            #batch_pred = torch.sigmoid(batch_out)

            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            # # In sigmoid it will return like [0.1, 0.9] where 0.1 is fake and 0.9 is genuine. However, we just care about fake value
            # # So we just take the first value and compare it with 0.5 to get the label
            # # If first value > 0.5 then label = 1 else label = 0
            
            
            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            num_correct += (predicted_labels == batch_y).sum().item()

        accuracy = (num_correct / num_total) * 100
        print("accuracy",accuracy)
        val_loss /= num_total
        # eval_accuracy = (num_correct / num_total) * 100
        print('[VALIDATION] eval_accuracy: ', accuracy)
        return val_loss, accuracy

def self_KD_teacher_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, student_forward_hook_manager, exp_lr_scheduler=None, temperature: float =3, alpha: float =0.1, beta: float = 1e-6,  use_amp: bool = True):
    logger.info('Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {}'.format(temperature, alpha, beta))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    running_total_hidden_rep_loss = 0
    
    
    student.train()
    if teacher is not None:
        teacher.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    cosine_loss = nn.CosineEmbeddingLoss()

    num_total = 0.0
    
    if not config['train']['teacher']:
        logger.info('No teacher')
        del teacher
    
    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        logger.info("Current learning rate: {}".format(exp_lr_scheduler.get_last_lr()[0]))
    else:
        logger.info("Current learning rate: {}".format(optimizer.param_groups[0]['lr']))
    iters = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (batch_x, batch_y) in pbar:
        
    
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            
            student_io_dict = student_forward_hook_manager.pop_io_dict()

            # Get teacher output
            if config['train']['teacher']:
                with torch.no_grad():
                    logits = teacher(batch_x)
                    teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

            batch_y = batch_y.view(-1).type(torch.int64).to(device)


            # Multiple loss
            total_mid_level_loss = 0

            # Check if key exists

            if 'criterions' in config and 'criterion_weights' in config:

                if len(config['criterions']) != len(config['criterion_weights']):
                    raise ValueError('Number of criterions and criterion_weights must be the same')

                # Hard code
                for loss, weight in zip(config['criterions'], config['criterion_weights']):
                    weight = float(weight)
                    loss_i = get_mid_level_loss(mid_level_criterion_config = loss)
                    total_mid_level_loss += (loss_i.forward(student_io_dict, teacher_io_dict) * weight)
                # teacher_hidden_representation = teacher_io_dict['LL']['output']
                # weight = float(config['criterion_weights'][0])
                # student_hidden_representation = student_hidden_representation.view(batch_size, -1)
                # teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)
                # hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))
                # total_mid_level_loss += (hidden_rep_loss * weight)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)
            graph_loss_S = criterion(graph_output_S, batch_y)
            graph_loss_T = criterion(graph_output_T, batch_y)
            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Calculate KD loss
            temp = batch_out / temperature
            temp = torch.softmax(temp, dim=1)
            temp_detach = temp.detach()
            kd_spectral_loss = kd_loss_function(spectral_output, temp_detach, temperature) * (temperature**2)
            kd_temporal_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_S = kd_loss_function(graph_output_S, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_T = kd_loss_function(graph_output_T, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_S = kd_loss_function(hs_gal_output_S, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_T = kd_loss_function(hs_gal_output_T, temp_detach, temperature) * (temperature**2)

            # Calculate loss (feature loss)
            # We didn't apply backward for final feature
            feature_loss_1 = feature_loss_function(middle_feature1, final_feature1.detach())
            feature_loss_2 = feature_loss_function(middle_feature2, final_feature2.detach())

            # Calculate total loss
            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T

            # Total KD loss
            total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_graph_loss_S + kd_graph_loss_T + kd_hs_gal_loss_S + kd_hs_gal_loss_T

            # Total feature loss
            total_feature_loss = feature_loss_1 + feature_loss_2

            # Total loss
            if 'criterions' in config and 'criterion_weights' in config:
                total_loss = (1 - alpha) * total_label_loss + (alpha * total_kd_loss) + (beta * total_feature_loss) +  total_mid_level_loss
            else:
                total_loss = (1 - alpha) * total_label_loss + (alpha * total_kd_loss) + (beta * total_feature_loss)

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] == 'ReduceLROnPlateau':
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()
        
        running_loss += (total_loss.item() * batch_size)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)
        
        if 'criterions' in config and 'criterion_weights' in config:
            running_total_hidden_rep_loss += (total_mid_level_loss.item() * batch_size)

    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss, running_total_hidden_rep_loss

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


def train_knowledge_distillation_config(cfg):
    args = Namespace(database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22', protocols_path='protocol.txt', batch_size=25, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=False, KD_cosine=False, KD_mse=False, self_KD=False, self_KD_teacher=True, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40, workers=8)
    

    if cfg['augment_mode'].startswith('raw'):
        args.algo = int(cfg['augment_mode'].split('_')[-1])
        cfg['augment_mode'] = 'rawboost'

    with open("/datab/hungdx/KDW2V-AASISTL/distill-config/trial17.yaml", 'r') as f:
        logger.info('Load configuration file {}'.format("/datab/hungdx/KDW2V-AASISTL/distill-config/trial17.yaml"))
        config = yaml.safe_load(f)

    seed = config['train']['seed']

    config['train']['learning_rate'] = cfg['lr']

    set_random_seed(seed, args)
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # teacher_model = get_model(config['model']['teacher']['name'], device=device).to(device)
    # student_model = get_model(config['model']['student']['name'], device=device).to(device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    student_model = SelfDistil_W2V2BASE_AASISTL(device).to(device)
    # teacher_model = W2V2_AASIST(device).to(device)

    teacher_forward_hook_manager = ForwardHookManager(device)
    student_forward_hook_manager = ForwardHookManager(device)

    student_model = torch.nn.DataParallel(student_model).to(device)
    # teacher_model = torch.nn.DataParallel(teacher_model).to(device)

    # if "is_parallel" in config["model"]["teacher"] and not config["model"]["teacher"]["is_parallel"]:
    #     logger.info("Teacher model is not parallel")
    #     teacher_model = teacher_model.to(device)
    # else:
    #     logger.info("Teacher model is parallel")
    #     teacher_model = torch.nn.DataParallel(teacher_model).to(device)

    # if "pretrained_path" in config["model"]["teacher"]:
        
    #     teacher_model.load_state_dict(torch.load(config["model"]["teacher"]["pretrained_path"],map_location=device))
    #     logger.info("Loaded teacher model from {}".format(config["model"]["teacher"]["pretrained_path"]))
    # else:
    #     teacher_model.load_state_dict(torch.load(args.model_path,map_location=device))
    #     logger.info("Loaded teacher model from {}".format(args.model_path))


    if "student_resume" in config["train"]:
        student_model.load_state_dict(torch.load(config["train"]["student_resume"],map_location=device))
        logger.info("Loaded student model from {}".format(config["train"]["student_resume"]))

    # Register forward hook
    logger.info('Register forward hook for teacher')
    for module_path, ios in zip(config['model']['teacher']['teacher_module_paths'], config['model']['teacher']['teacher_module_ios']):
        logger.info('Register forward hook for {}'.format(module_path))
        requires_input, requires_output =  ios.split(':')
        requires_input, requires_output = bool(requires_input), bool(requires_output)
        # if "is_parallel" in config["model"]["teacher"] and not config["model"]["teacher"]["is_parallel"]:
        #     teacher_forward_hook_manager.add_hook(teacher_model, module_path, requires_input=requires_input, requires_output=requires_output)
        # else:
        #     teacher_forward_hook_manager.add_hook(teacher_model.module, module_path, requires_input=requires_input, requires_output=requires_output)

    logger.info('Register forward hook for student')
    for module_path, ios in zip(config['model']['student']['student_module_paths'], config['model']['student']['student_module_ios']):
        logger.info('Register forward hook for {}'.format(module_path))
        requires_input, requires_output =  ios.split(':')
        requires_input, requires_output = bool(requires_input), bool(requires_output)
        student_forward_hook_manager.add_hook(student_model.module, module_path, requires_input=requires_input, requires_output=requires_output)


    logger.info('Prepare training, dev set .....')
    print(config["train"]["dataset"])

    
    if "augment_mode" in config["train"]:
        config["train"]["augment_mode"] = cfg['augment_mode']    
        logger.info(f'Use new data augmentation {config["train"]["augment_mode"]}')
        if "dataset" in config["train"]:
            logger.info(f'Use {config["train"]["dataset"]} dataset')
            train_loader, dev_loader = get_train_dev_dataloader(args, config["train"]["augment_mode"], dataset=config["train"]["dataset"])
        else:
            train_loader, dev_loader = get_train_dev_dataloader(args, config["train"]["augment_mode"])
    else:
        logger.info('Use RawBoots data augmentation')
        if "dataset" in config["train"]:
            logger.info(f'Use {config["train"]["dataset"]} dataset')
            train_loader, dev_loader = get_train_dev_dataloader(args, dataset=config["train"]["dataset"])
        else:
            train_loader, dev_loader = get_train_dev_dataloader(args)


    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])

    exp_lr_scheduler = None
    if 'is_learning_rate_scheduler' in config and config['is_learning_rate_scheduler']:
        logger.info(f'Use learning rate scheduler {config["learning_rate_scheduler"]["name"]}')
        # Initialize learning rate scheduler by using its name and its parameters
        exp_lr_scheduler = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler']['name'])(optimizer, **config['learning_rate_scheduler']['params'])

    scaler = torch.cuda.amp.GradScaler(enabled=config['train']['amp'])

    best_val_loss = None
        
    early_stopping = EarlyStopping(patience=30, verbose=True, model_save_path='./')
        
    for epoch in range(MAX_EPOCHS):
           
            ## Self KD
        
        train_loss, train_total_label_loss, train_total_kd_loss, train_total_feature_loss, running_total_hidden_rep_loss = self_KD_teacher_train_epoch(train_loader, student_model, None, optimizer, device, scaler, config, student_forward_hook_manager, exp_lr_scheduler,  temperature=cfg['temperature'], alpha=cfg['alpha'], beta=cfg['beta'], use_amp = True)
        eval_loss, accuracy = self_KD_teacher_val_epoch(dev_loader, student_model, device)



        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            early_stopping.model_save_path = temp_checkpoint_dir
            path = os.path.join(temp_checkpoint_dir, "best_model.pth")
       
            if best_val_loss is None or eval_loss < best_val_loss:
                best_val_loss = eval_loss
                torch.save(student_model.state_dict(), path)
                
            # early_stopping(eval_loss, student_model, epoch)

            # if early_stopping.early_stop:
            #     logging.log(logging.INFO, "Early stopping")
            #     break


        # # # Report both training and validation loss to Tune
        metrics = {'train_loss': train_loss, 'val_loss': eval_loss}
        # Report both training and validation loss to Tune
        ray.train.report(metrics)

def kd_train_epoch(train_loader, student, optimizer, device, scaler, config, criterion_list, method, exp_lr_scheduler=None, use_amp: bool = True):
    logger.info('self KD training')
    running_loss = 0
    running_loss_cls = 0
    running_loss_div = 0
    student.train()
    
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    num_total = 0.0
  
    
    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        logger.info("Current learning rate: {}".format(exp_lr_scheduler.get_last_lr()[0]))
    else:
        logger.info("Current learning rate: {}".format(optimizer.param_groups[0]['lr']))
    
    num_classes = 2
    iters = len(train_loader)
    # Create a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (batch_x, batch_y) in pbar:
        
    
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # Multiple loss
            loss_div = torch.tensor(0.).to(device)
            loss_cls = torch.tensor(0.).to(device)

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            

            if method == 'cross_entropy':
                logit = student(batch_x)
                loss_cls += criterion_cls(logit, batch_y)
            elif method == 'mixup':   
                logit, mixup_loss = Mixup(student, batch_x, batch_y, criterion_cls, alpha=0.4)
                loss_cls += mixup_loss
            elif method == 'manifold_mixup':   
                logit, manifold_mixup_loss = ManifoldMixup(student, batch_x, batch_y, criterion_cls, alpha=2.0)
                loss_cls += manifold_mixup_loss
            elif method == 'cutmix':   
                logit, cutmix_loss = CutMix(student, batch_x, batch_y, criterion_cls, alpha=1.0)
                loss_cls += cutmix_loss
            elif method == 'label_smooth':
                logit = student(batch_x)
                loss_cls += LabelSmooth(logit, batch_y, num_classes=num_classes)
            elif method == 'FocalLoss':
                logit = student(batch_x) 
                loss_cls += FocalLoss(logit, batch_y)
            elif method == 'TF_KD_self_reg':
                logit = student(batch_x)
                loss_cls += criterion_cls(logit, batch_y)
                loss_div += TF_KD_reg(logit, batch_y, num_classes, epsilon=0.1, T=20)
            elif method == 'virtual_softmax':
                logit = student(batch_x, batch_y, loss_type='virtual_softmax')
                loss_cls += criterion_cls(logit, batch_y)
            elif method == 'Maximum_entropy':
                logit = student(batch_x, batch_y)
                entropy = (F.softmax(logit, dim=1) * F.log_softmax(logit, dim=1)).mean()
                loss_cls += criterion_cls(logit, batch_y) + 0.5 * entropy

            elif method == 'DKS':
                logit, dks_loss_cls, dks_loss_div = DKS(student, batch_x, batch_y, criterion_cls, criterion_div)
                loss_cls += dks_loss_cls
                loss_div += dks_loss_div

            elif method == 'SAD':
                logit, sad_loss_cls, sad_loss_div = SAD(student, batch_x, batch_y, criterion_cls, criterion_div)
                loss_cls += sad_loss_cls
                loss_div += sad_loss_div
            
            elif method == 'DDGSD':
                logit, ddsgd_loss_cls, ddsgd_loss_div = DDGSD(student, batch_x, batch_y, criterion_cls, criterion_div)
                loss_cls += ddsgd_loss_cls
                loss_div += ddsgd_loss_div

            elif method == 'CS-KD':
                logit, cs_kd_loss_cls, cs_kd_loss_div = CS_KD(student, batch_x, batch_y, criterion_cls, criterion_div)
                batch_y = batch_y[:batch_size//2]
                batch_size = batch_size // 2
                loss_cls += cs_kd_loss_cls
                loss_div += cs_kd_loss_div
                
            elif method.startswith('FRSKD'):
                logit, frskd_loss_cls, frskd_loss_div = FRSKD(student, batch_x, batch_y, criterion_cls, criterion_div)
                loss_cls += frskd_loss_cls
                loss_div += frskd_loss_div

            elif method.startswith('BAKE'):
                logit, bake_loss_cls, bake_loss_div = BAKE(student, batch_x, batch_y, criterion_cls, criterion_div, args)
                loss_cls += bake_loss_cls
                loss_div += bake_loss_div
        
            else:
                raise ValueError('Unknown method: {}'.format(args.method))
            
            loss = loss_cls + loss_div

        # Scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update LR
        if exp_lr_scheduler is not None:
            name_scheduler = config['learning_rate_scheduler']['name']
            if name_scheduler == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif name_scheduler in ['ReduceLROnPlateau', 'MultiStepLR']:
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        running_loss_cls += (loss_cls.item() * batch_size)
        running_loss_div += (loss_div.item() * batch_size)
        

    running_loss /= num_total
    running_loss_cls /= num_total
    running_loss_div /= num_total
    
    return running_loss, running_loss_cls, running_loss_div

def kd_val_epoch(dev_loader, model, device, criterion_cls):
    logger.info('Validation ----')
    val_loss = 0
    model.eval()

    num_total = 0.0
    num_correct = 0.0

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            
            batch_out = model(batch_x)
            
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion_cls(batch_out, batch_y)
        
            val_loss += (batch_loss.item() * batch_size)

            probabilities = F.softmax(batch_out, dim=1)
            predicted_labels = (probabilities[:,0] >= 0.5).int()

            num_correct += (predicted_labels == batch_y).sum().item()
            

        accuracy = (num_correct / num_total) * 100
        print("accuracy",accuracy)
        val_loss /= num_total
        print('[VALIDATION] eval_accuracy: ', accuracy)
        return val_loss, accuracy



def train_self_knowledge_distillation_config(cfg):
    args = Namespace(database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22', protocols_path='protocol.txt', batch_size=25, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=False, KD_cosine=False, KD_mse=False, self_KD=False, self_KD_teacher=True, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40, workers=8)
    

    if cfg['augment_mode'].startswith('raw'):
        args.algo = int(cfg['augment_mode'].split('_')[-1])
        cfg['augment_mode'] = 'rawboost'

    with open("/datab/hungdx/KDW2V-AASISTL/self-kd-config/trial1.yaml", 'r') as f:
        logger.info('Load configuration file {}'.format("/datab/hungdx/KDW2V-AASISTL/self-kd-config/trial1.yaml"))
        config = yaml.safe_load(f)

    seed = config['train']['seed']

    config['train']['learning_rate'] = cfg['lr']
    use_amp = config['train'].get('amp', False)
    set_random_seed(seed, args)
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

   
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    student = Distil_W2V2BASE_AASISTL(device).to(device)

    student = torch.nn.DataParallel(student).to(device)
   
    if "student_resume" in config["train"]:
        student.load_state_dict(torch.load(config["train"]["student_resume"],map_location=device))
        logger.info("Loaded student model from {}".format(config["train"]["student_resume"]))

    sup_contrastive = config['train'].get('sup_contrastive', False)
    augment_mode = config["train"].get("augment_mode", "rawboost")
    dataset = config["train"].get("dataset", "LA19")
    T = config["train"].get("T", 1.0)
    method = config["train"].get("method", "cross_entropy")
    ce_weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)

    if sup_contrastive:
        train_loader, dev_loader = get_train_dev_dataloader_contrastive(args)
    else:
        train_loader, dev_loader = get_train_dev_dataloader(args, augment_mode, dataset)

    optimizer = torch.optim.Adam(student.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'])


    exp_lr_scheduler = None
    if 'is_learning_rate_scheduler' in config and config['is_learning_rate_scheduler']:
        logger.info(f'Use learning rate scheduler {config["learning_rate_scheduler"]["name"]}')
        # Initialize learning rate scheduler by using its name and its parameters
        exp_lr_scheduler = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler']['name'])(optimizer, **config['learning_rate_scheduler']['params'])

    scaler = torch.cuda.amp.GradScaler(enabled=config['train']['amp'])
    criterion_cls = nn.CrossEntropyLoss(ce_weight)
    criterion_div = DistillKL(T)
    criterion_list = nn.ModuleList([])

    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)
    criterion_list.to(device)

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    

    for epoch in range(MAX_EPOCHS):
        train_loss, train_loss_cls, train_loss_div = kd_train_epoch(train_loader, student, optimizer, device, scaler, config, criterion_list, method, exp_lr_scheduler, use_amp)
        # Eval
        eval_loss, accuracy = kd_val_epoch(dev_loader, student, device, criterion_cls)

        # # # Report both training and validation loss to Tune
        metrics = {'train_loss': train_loss, 'val_loss': eval_loss}
        # Report both training and validation loss to Tune
        ray.train.report(metrics)

def test_best_model(best_result, model_type="KD_base_cosine"):
    
    args = Namespace(database_path='/datab/hungdx/KDW2V-AASISTL/databases/', protocols_path='/datab/hungdx/KDW2V-AASISTL/protocols/', batch_size=64, num_epochs=100, lr=1e-06, weight_decay=0.0001, loss='weighted_CCE', seed=1234, model_path='/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-teacher.pth', cudnn_deterministic_toggle=True, cudnn_benchmark_toggle=False, student_restore=False, KD_logits=False, KD_cosine=False, KD_mse=True, algo=3, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)
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
    "temperature": tune.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]),
    "alpha": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "beta": tune.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "lr": tune.choice([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]),
    "weight_decay": tune.choice([0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]),
    "augment_mode": tune.choice(["audiomentations", "rawboost_1", "rawboost_2", "rawboost_3", "rawboost_4", "rawboost_5"]),
    
}

SAMPLES = 1
for key, value in search_space.items():
    SAMPLES = SAMPLES * len(value)

print("Total number of samples: ", SAMPLES)

# Define the scheduler

# Create a stopper object
# stopper = ExperimentPlateauStopper(metric="val_loss", mode="min", patience =10, top=2)


# Run the hyperparameter search

# analysis = tune.run(
#     train_knowledge_distillation_config, 
#     config=search_space, 
#     num_samples=SAMPLES,  # Number of samples to run
#     resources_per_trial={"cpu": 1, "gpu": 0.5},  # Resources to allocate per trial
#     # stop = stopper
#     resume="AUTO",
#     max_concurrent_trials=8,
# )

# # Get the best configuration
# best_config = analysis.get_best_config(metric="val_loss", mode="min")
# print(best_config)
class CustomStopper(Stopper):
    def __init__(self):
        self.should_stop = False
        self.patience = 10
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, trial_id: str, result: dict) -> bool:
        """Returns whether to stop the given trial."""
        if not self.should_stop and result["val_loss"] <= 0.001 and result["train_loss"] <= 0.001:
            self.should_stop = True
        """ Early stopping if the val_loss is not improving for the last 10 epochs"""
        
        score = result["val_loss"]
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop

    def stop_all(self) -> bool:
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop

NUM_SAMPLES = 300
MAX_CONCURRENT = 4
algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
algo = ConcurrencyLimiter(algo, max_concurrent=MAX_CONCURRENT)
scheduler = AsyncHyperBandScheduler()

stopper = CustomStopper()


tuner = tune.Tuner(
        train_self_knowledge_distillation_config,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=NUM_SAMPLES,
            
        ),
        
        param_space={
            "lr": tune.grid_search([1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]),
            "weight_decay": tune.grid_search([0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001])
        },

        run_config=RunConfig(storage_path="./ray_results", name="test_experiment", 
                                log_to_file=("my_stdout.log", "my_stderr.log"), stop=stopper,
                                checkpoint_config=train.CheckpointConfig(
                                                checkpoint_score_attribute="val_loss",
                                                num_to_keep=5)
                            )
                    
    )
    
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result().config)