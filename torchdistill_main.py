import torch
import os
import sys
from torchdistill.models.registry import get_model
from torchdistill.losses.registry import get_mid_level_loss
from torchdistill.core.forward_hook import ForwardHookManager
from data_utils import *
from student import *
from teacher import *
from torchdistill_utils import *
import yaml
from startup_config import set_random_seed
from menu import get_main_menu
from main import get_train_dev_dataloader
from kdtoolkit import kd_loss_function, feature_loss_function
from utils import EarlyStopping
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
from main import W2V2_TA
import wandb
from datetime import timedelta
from wandb import AlertLevel
from contrast.supcontrastloss import SupConLoss
from engine.dot import DistillationOrientedTrainer
import eval_metrics_DF as em

# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""

#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t):
#         p_s = F.log_softmax(y_s / self.T, dim=1)
#         p_t = F.softmax(y_t / self.T, dim=1)
#         loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
#         return loss
# from neural_compressor.training import prepare_compression

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Device: {}'.format(device))

# def self_KD_teacher_val_epoch(dev_loader, model, device):
#     logger.info('Validation Teacher self KD')
#     val_loss = 0
#     model.eval()
#     weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weight)
#     num_total = 0.0
#     num_correct = 0.0

#     with torch.inference_mode():
#         for batch_x, batch_y in tqdm(dev_loader):
#             batch_size = batch_x.size(0)
#             num_total += batch_size
#             batch_x = batch_x.to(device)
            
#             batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            
#             batch_y = batch_y.view(-1).type(torch.int64).to(device)

#             # Calculate loss (label loss)
#             batch_loss = criterion(batch_out, batch_y)
#             spectral_loss = criterion(spectral_output, batch_y)
#             temporal_loss = criterion(temporal_output, batch_y)
#             graph_loss_S = criterion(graph_output_S, batch_y)
#             graph_loss_T = criterion(graph_output_T, batch_y)
#             hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
#             hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

#             # Total label loss
#             total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T
            
#             total_loss = total_label_loss

        
#             val_loss += (total_loss.item() * batch_size)

#             probabilities = F.softmax(batch_out, dim=1)
#             predicted_labels = (probabilities[:,0] >= 0.5).int()

#             # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
#             num_correct += (predicted_labels == batch_y).sum().item()

#         accuracy = (num_correct / num_total) * 100
#         print("accuracy",accuracy)
#         val_loss /= num_total
#         # eval_accuracy = (num_correct / num_total) * 100
#         print('[VALIDATION] eval_accuracy: ', accuracy)
#         return val_loss, accuracy

# def self_KD_teacher_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, exp_lr_scheduler=None, temperature: float =3, alpha: float =0.1, beta: float = 1e-6,  use_amp: bool = True):
#     logger.info('Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {}'.format(temperature, alpha, beta))
#     running_loss = 0
#     running_total_label_loss = 0
#     running_total_kd_loss = 0
#     running_total_feature_loss = 0
#     running_total_hidden_rep_loss = 0
#     running_sup_contrastive_loss = 0
    
#     student.train()
#     teacher.eval()
#     weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weight)

#     num_total = 0.0
    
#     if not config['train']['teacher']:
#         logger.info('No teacher')
#         del teacher
    
#     if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
#         logger.info("Current learning rate: {}".format(exp_lr_scheduler.get_last_lr()[0]))
#     else:
#         logger.info("Current learning rate: {}".format(optimizer.param_groups[0]['lr']))
    
#     iters = len(train_loader)
#     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#     for i, (batch_x, batch_y) in pbar:
         
#         # Mixed precision training
#         with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            

#             batch_size = batch_x.size(0)
#             num_total += batch_size
#             batch_x = batch_x.to(device)
#             batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            
#             student_io_dict = student_forward_hook_manager.pop_io_dict()

#             # Get teacher output
#             if config['train']['teacher']:
#                 with torch.no_grad():
#                     logits = teacher(batch_x)
#                     teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

#             batch_y = batch_y.view(-1).type(torch.int64).to(device)


#             # Multiple loss
#             total_mid_level_loss = 0

#             # Check if key exists

#             if 'criterions' in config and 'criterion_weights' in config:

#                 if len(config['criterions']) != len(config['criterion_weights']):
#                     raise ValueError('Number of criterions and criterion_weights must be the same')

#                 # Hard code
#                 for loss, weight in zip(config['criterions'], config['criterion_weights']):
#                     weight = float(weight)
#                     # print(loss)
#                     # if loss == 'OCKDLoss':
#                     #     # Disable dropout
                    
#                     loss_i = get_mid_level_loss(mid_level_criterion_config = loss)
#                     total_mid_level_loss += (loss_i.forward(student_io_dict, teacher_io_dict) * weight)

#             # Calculate loss (label loss)
#             batch_loss = criterion(batch_out, batch_y)
#             spectral_loss = criterion(spectral_output, batch_y)
#             temporal_loss = criterion(temporal_output, batch_y)
#             graph_loss_S = criterion(graph_output_S, batch_y)
#             graph_loss_T = criterion(graph_output_T, batch_y)
#             hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
#             hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

#             # Calculate KD loss
#             temp = batch_out / temperature
#             temp = torch.softmax(temp, dim=1)
#             temp_detach = temp.detach()
#             kd_spectral_loss = kd_loss_function(spectral_output, temp_detach, temperature) * (temperature**2)
#             kd_temporal_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
#             kd_graph_loss_S = kd_loss_function(graph_output_S, temp_detach, temperature) * (temperature**2)
#             kd_graph_loss_T = kd_loss_function(graph_output_T, temp_detach, temperature) * (temperature**2)
#             kd_hs_gal_loss_S = kd_loss_function(hs_gal_output_S, temp_detach, temperature) * (temperature**2)
#             kd_hs_gal_loss_T = kd_loss_function(hs_gal_output_T, temp_detach, temperature) * (temperature**2)

#             # Calculate loss (feature loss)
#             # We didn't apply backward for final feature
#             feature_loss_1 = feature_loss_function(middle_feature1, final_feature1.detach())
#             feature_loss_2 = feature_loss_function(middle_feature2, final_feature2.detach())

#             # Calculate total loss
#             # Total label loss
#             total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T

#             # Total KD loss
#             total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_graph_loss_S + kd_graph_loss_T + kd_hs_gal_loss_S + kd_hs_gal_loss_T

#             # Total feature loss
#             total_feature_loss = feature_loss_1 + feature_loss_2

#             # Total loss
#             if 'criterions' in config and 'criterion_weights' in config:
#                 total_loss = (1 - alpha) * total_label_loss + (alpha * total_kd_loss) + (beta * total_feature_loss) +  total_mid_level_loss
#             else:
#                 total_loss = (1 - alpha) * total_label_loss + (alpha * total_kd_loss) + (beta * total_feature_loss)
            
#             if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
#                 sup_weights = config["train"].get("sup_contrastive_loss_weight", [0.07, 0.07])
#                 sup_loss = SupConLoss(temperature=sup_weights[0], base_temperature=sup_weights[1])
#                 sup_mode = config["train"].get("sup_contrastive_mode", "supcon")

#                 if sup_mode == "supcon":
#                     sup_loss = sup_loss.forward(student_hidden_representation, batch_y)
#                 else:
#                     sup_loss = sup_loss.forward(student_hidden_representation)
                
#                 total_loss += sup_loss

#         # Scaler
#         optimizer.zero_grad()
#         scaler.scale(total_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         # Update LR
#         if exp_lr_scheduler is not None:
#             if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
#                 exp_lr_scheduler.step(epoch + i / iters)
#             # elif config['learning_rate_scheduler']['name'] in ['ReduceLROnPlateau', 'MultiStepLR']:
#             #     # Update learning rate scheduler in validation so do nothing here
#             #     pass
#             # else:
#             #     exp_lr_scheduler.step()
        
#         running_loss += (total_loss.item() * batch_size)
#         running_total_label_loss += (total_label_loss.item() * batch_size)
#         running_total_kd_loss += (total_kd_loss.item() * batch_size)
#         running_total_feature_loss += (total_feature_loss.item() * batch_size)
        
#         if 'criterions' in config and 'criterion_weights' in config:
#             running_total_hidden_rep_loss += (total_mid_level_loss.item() * batch_size)
#         if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
#             running_sup_contrastive_loss += (sup_loss.item() * batch_size)
#     running_loss /= num_total
#     running_total_feature_loss /= num_total
#     running_total_label_loss /= num_total
#     running_total_kd_loss /= num_total
#     return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss

# def kd_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, exp_lr_scheduler=None,  use_amp: bool = True):
#     logger.info('Training KD')
#     running_loss = 0
    
#     student.train()
#     teacher.eval()

#     if "alpha" not in config['train']:
#         forward_target = True
#     else:
#         forward_target = False
#         alpha = float(config['train']['alpha'])

#         if 'beta' in config['train']:
#             beta = float(config['train']['beta'])
#         else:
#             beta = 0.5
  
#     weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
#     criterion = nn.CrossEntropyLoss(weight=weight)
#     dot = config['train'].get('dot', False)

#     num_total = 0.0
    
#     if not config['train']['teacher']:
#         logger.info('No teacher')
#         del teacher
    
#     if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
#         logger.info("Current learning rate: {}".format(exp_lr_scheduler.get_last_lr()[0]))
#     else:
#         logger.info("Current learning rate: {}".format(optimizer.param_groups[0]['lr']))
        
#     iters = len(train_loader)
#     # Create a progress bar
#     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#     for i, (batch_x, batch_y) in pbar:
        
    
#         # Mixed precision training
#         with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
#             # Multiple loss
#             total_loss = torch.tensor(0.).to(device)
#             kd_loss = torch.tensor(0.).to(device)
#             ce_loss = torch.tensor(0.).to(device)

#             batch_size = batch_x.size(0)
#             num_total += batch_size
#             batch_x = batch_x.to(device)
#             batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            
#             student_io_dict = student_forward_hook_manager.pop_io_dict()

#             # Get teacher output
#             if config['train']['teacher']:
#                 with torch.no_grad():
#                     t_logits = teacher(batch_x)
#                     teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

#             batch_y = batch_y.view(-1).type(torch.int64).to(device)


            

#             # Check if key exists

#             if 'criterions' in config and 'criterion_weights' in config:

#                 if len(config['criterions']) != len(config['criterion_weights']):
#                     raise ValueError('Number of criterions and criterion_weights must be the same')

#                 for loss, weight in zip(config['criterions'], config['criterion_weights']):
#                     weight = float(weight)
#                     # print("Current loss: ", loss)
#                     if loss['key'] == 'OCKDLoss':
#                         student.module.ssl_model.model.eval()
#                     else:
#                         student.module.ssl_model.model.train()
                    
#                     loss_i = get_mid_level_loss(mid_level_criterion_config = loss)
                    

#                     if forward_target:
#                         total_loss += (loss_i.forward(student_io_dict, teacher_io_dict, batch_y) * weight)
#                         kd_loss += (loss_i.forward(student_io_dict, teacher_io_dict, batch_y) * weight)
#                     else:
#                         total_loss += (loss_i.forward(student_io_dict, teacher_io_dict) * weight)
#                         kd_loss += (loss_i.forward(student_io_dict, teacher_io_dict) * weight)


#             if not forward_target or dot:
#                 alpha = 1
#                 # CE loss
#                 batch_loss = criterion(batch_out, batch_y)
#                 total_loss +=  (alpha * batch_loss)
#                 ce_loss += (alpha * batch_loss)

#                 # ## Total loss + KL divergence loss
#                 # kl_loss = DistillKL(T=config["train"].get("T", 2))
#                 # total_loss += (beta * kl_loss(batch_out, t_logits))
            
            
#             if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
#                 sup_weights = config["train"].get("sup_contrastive_loss_weight", [0.1, 0.07])
#                 sup_loss = SupConLoss(temperature=sup_weights[0], base_temperature=sup_weights[1])
#                 sup_mode = config["train"].get("sup_contrastive_mode", "supcon")

#                 if sup_mode == "supcon":
#                     sup_loss = sup_loss.forward(student_hidden_representation, batch_y)
#                 else:
#                     sup_loss = sup_loss.forward(student_hidden_representation)
#                 total_loss += sup_loss

#         # Scaler
#         if not dot:
#             optimizer.zero_grad()
#             scaler.scale(total_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             optimizer.zero_grad(set_to_none=True)
            
#             kd_loss.backward(retain_graph=True)
#             optimizer.step_kd()
#             optimizer.zero_grad(set_to_none=True)
#             ce_loss.backward()
#             optimizer.step()
#         # Update LR
#         if exp_lr_scheduler is not None:
#             if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
#                 exp_lr_scheduler.step(epoch + i / iters)
#             elif config['learning_rate_scheduler']['name'] == 'ReduceLROnPlateau' or config['learning_rate_scheduler']['name'] == 'MultiStepLR' or config['learning_rate_scheduler']['name'] == 'StepLR':
#                 # Update learning rate scheduler in validation so do nothing here
#                 pass
#             else:
#                 exp_lr_scheduler.step()
#         if not dot:
#             running_loss += (total_loss.item() * batch_size)
#         else:
#             # logger.info("KD loss: {} - CE loss: {}".format(kd_loss.item(), ce_loss.item()))
#             running_loss += (ce_loss.item() + kd_loss.item()) * batch_size
            


#     running_loss /= num_total
    
#     return running_loss

# def kd_val_epoch(dev_loader, model, device):
#     logger.info('Validation ----')
#     val_loss = 0
#     model.eval()    
#     weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    
#     criterion = nn.CrossEntropyLoss(weight=weight)
#     num_total = 0.0
#     num_correct = 0.0
#     bona_scores = []
#     spoof_scores = []

#     with torch.inference_mode():
#         for batch_x, batch_y in tqdm(dev_loader):
#             batch_size = batch_x.size(0)
#             num_total += batch_size
#             batch_x = batch_x.to(device)
            
#             batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            
#             batch_y = batch_y.view(-1).type(torch.int64).to(device)

#             # Calculate loss (label loss)
#             batch_loss = criterion(batch_out, batch_y)
        
#             val_loss += (batch_loss.item() * batch_size)

#             probabilities = F.softmax(batch_out, dim=1)
#             predicted_labels = (probabilities[:,0] >= 0.5).int()

#             num_correct += (predicted_labels == batch_y).sum().item()

#             # Calculate EER
#             # for x, y in zip(batch_x, batch_y):
#             #     if y == 1:
#             #         bona_scores.append(x)
#             #     else:
#             #         spoof_scores.append(x)
            
#         # eer_cm, th = em.compute_eer(bona_scores, spoof_scores) * 100
#         # logger.info("EER: {}% - Threshold: {}".format(eer_cm , th))
#         accuracy = (num_correct / num_total) * 100
#         print("accuracy",accuracy)
#         val_loss /= num_total
#         print('[VALIDATION] eval_accuracy: ', accuracy)
#         return val_loss, accuracy

def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, config, exp_lr_scheduler=None, use_amp: bool = True):
    running_loss = 0
    model.train()
    num_total = 0.0
    iters = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (batch_x, batch_y) in pbar:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            loss = criterion(batch_out, batch_y)
            running_loss += (loss.item() * batch_size)
        # Scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] in ['ReduceLROnPlateau', 'MultiStepLR']:
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()
    running_loss /= num_total
    return running_loss

def validation(dev_loader, model, criterion, device):
    model.eval()
    num_total = 0.0
    num_correct = 0.0
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            loss = criterion(batch_out, batch_y)
            val_loss += (loss.item() * batch_size)
            probabilities = F.softmax(batch_out, dim=1)
            predicted_labels = (probabilities[:,0] >= 0.5).int()
            num_correct += (predicted_labels == batch_y).sum().item()
    val_loss /= num_total
    accuracy = (num_correct / num_total) * 100

    return val_loss, accuracy

args = get_main_menu()
# Load configuration
with open(args.yaml, 'r') as f:
    logger.info('Load configuration file {}'.format(args.yaml))
    config = yaml.safe_load(f)


seed = config['train'].get('seed', 1234)
ce_weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
student_resume_path = config['train'].get('student_resume', None)
sup_contrastive = config['train'].get('sup_contrastive', False)
is_learning_rate_scheduler = config['train'].get('is_learning_rate_scheduler', False)
patience = config['train'].get('patience', 10)
use_amp = config['train'].get('amp', False)
learning_rate_scheduler_name = config['learning_rate_scheduler'].get('name', None)
student_model_name = config['model']['student'].get('name', 'Distil_W2V2BASE_AASISTL')
teacher_model_name = config['model']['teacher'].get('name', 'W2V2_TA')
model_path = config['model']['teacher'].get('pretrained_path', None)
student_model_path = config['train'].get('student_resume', None)
augment_mode = config["train"].get("augment_mode", "rawboost")
dataset = config["train"].get("dataset", "LA19")
dot = config["train"].get("dot", False)

train_teacher = config["train"].get("train_teacher", False)
ssl_teacher_path = config["train"].get("ssl_teacher_path", "/nfs/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt")
ssl_student_path = config["train"].get("ssl_student_path", "/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

if augment_mode == "rawboost":
    # DEFAULT rawboost 3
    args.algo = config["train"].get("algo", 3)

# Wandb
args.batch_size = config['train'].get('batch_size', 32)
wandb.init(project="torchdistill", config={
    **config
}, name=config['name'])

set_random_seed(seed, args)
logger.info('Random seed: {}'.format(seed))


teacher_model = get_model(config['model']['teacher']['name'], device=device, ssl_cpkt_path=ssl_teacher_path).to(device)
student_model = get_model(config['model']['student']['name'], device=device, ssl_cpkt_path=ssl_student_path).to(device)

teacher_forward_hook_manager = ForwardHookManager(device)
student_forward_hook_manager = ForwardHookManager(device)

student_model = torch.nn.DataParallel(student_model).to(device)

if "is_parallel" in config["model"]["teacher"] and not config["model"]["teacher"]["is_parallel"]:
    logger.info("Teacher model is not parallel")
    teacher_model = teacher_model.to(device)
else:
    logger.info("Teacher model is parallel")
    teacher_model = torch.nn.DataParallel(teacher_model).to(device)

if "pretrained_path" in config["model"]["teacher"]:
    
    teacher_model.load_state_dict(torch.load(config["model"]["teacher"]["pretrained_path"],map_location=device))
    logger.info("Loaded teacher model from {}".format(config["model"]["teacher"]["pretrained_path"]))
else:
    teacher_model.load_state_dict(torch.load(args.model_path,map_location=device))
    logger.info("Loaded teacher model from {}".format(args.model_path))


if "student_resume" in config["train"]:
    student_model.load_state_dict(torch.load(config["train"]["student_resume"],map_location=device))
    logger.info("Loaded student model from {}".format(config["train"]["student_resume"]))

# Register forward hook
logger.info('Register forward hook for teacher')
for module_path, ios in zip(config['model']['teacher']['teacher_module_paths'], config['model']['teacher']['teacher_module_ios']):
    logger.info('Register teacher forward hook for {}'.format(module_path))
    requires_input, requires_output =  ios.split(':')
    requires_input, requires_output = bool(requires_input), bool(requires_output)
    if "is_parallel" in config["model"]["teacher"] and not config["model"]["teacher"]["is_parallel"]:
        teacher_forward_hook_manager.add_hook(teacher_model, module_path, requires_input=requires_input, requires_output=requires_output)
    else:
        teacher_forward_hook_manager.add_hook(teacher_model.module, module_path, requires_input=requires_input, requires_output=requires_output)

logger.info('Register forward hook for student')
for module_path, ios in zip(config['model']['student']['student_module_paths'], config['model']['student']['student_module_ios']):
    logger.info('Register student forward hook for {}'.format(module_path))
    requires_input, requires_output =  ios.split(':')
    requires_input, requires_output = bool(requires_input), bool(requires_output)
    student_forward_hook_manager.add_hook(student_model.module, module_path, requires_input=requires_input, requires_output=requires_output)


logger.info('Prepare training, dev set .....')

logger.info(f'Use {augment_mode} data augmentation')
if dataset:
    logger.info(f'Use {dataset} dataset')


if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
    logger.info('Use supervised contrastive learning')
    train_loader, dev_loader = get_train_dev_dataloader_contrastive(args)
else:
    train_loader, dev_loader = get_train_dev_dataloader(args, augment_mode, dataset)

if not dot:
    optimizer = torch.optim.Adam(student_model.parameters(), lr=float(config['train']['learning_rate']),weight_decay=config['train']['weight_decay'])
else:
    '''
        Initialize optimizer for Distillation-Oriented Trainer
    '''
    momentum = float(config['train'].get('momentum', 0.9))
    delta = float(config['train'].get('delta', 0.0075))
    m_task = momentum - delta
    m_kd = momentum + delta
    optimizer = DistillationOrientedTrainer(student_model.parameters(), lr=float(config['train']['learning_rate']), momentum=m_task, momentum_kd=m_kd, weight_decay=config['train']['weight_decay'])
    logger.info('Use Distillation-Oriented Trainer with momentum = {} and momentum_kd = {}'.format(m_task, m_kd))

exp_lr_scheduler = None
if 'is_learning_rate_scheduler' in config and config['is_learning_rate_scheduler']:
    logger.info(f'Use learning rate scheduler {config["learning_rate_scheduler"]["name"]}')
    # Initialize learning rate scheduler by using its name and its parameters
    exp_lr_scheduler = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler']['name'])(optimizer, **config['learning_rate_scheduler']['params'])
else:
    logger.info('No learning rate scheduler')



scaler = torch.cuda.amp.GradScaler(enabled=config['train']['amp'])
writer = SummaryWriter('logs/{}'.format(config['name']))
model_save_path = os.path.join("models", config['name'])

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    logger.info('Created model save path {}'.format(model_save_path))


early_stopping = EarlyStopping(patience=config['train']['patience'], verbose=True, model_save_path=model_save_path)

# Train loop
logger.info("Start training")
num_epochs = config['train']['num_epochs']

if train_teacher:
    logger.info('Train teacher model')
    weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for epoch in tqdm(range(num_epochs), colour='green'):
        train_loss = train_one_epoch(train_loader, teacher_model, criterion, optimizer, device, scaler, config, exp_lr_scheduler, use_amp = use_amp)
        eval_loss, accuracy = validation(dev_loader, teacher_model, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/eval', eval_loss, epoch)

        if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
            writer.add_scalar('Lr/epoch', exp_lr_scheduler.get_last_lr()[0], epoch)
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "learning_rate": exp_lr_scheduler.get_last_lr()[0]})
        else:
            writer.add_scalar('Lr/epoch', optimizer.param_groups[0]['lr'], epoch)
            wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "learning_rate": optimizer.param_groups[0]['lr']})

        if train_loss < 0.001 and eval_loss < 0.001:
            wandb.alert(
                title='Low loss',
                text=f'train_loss {train_loss} and eval_loss: {eval_loss} is below the acceptable threshold 0.001',
                level=AlertLevel.WARN,
                wait_duration=timedelta(minutes=5)
            )
        # Early stopping
        early_stopping(eval_loss, student_model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        # Save model
        if epoch % 1 == 0:
            torch.save(student_model.state_dict(), os.path.join(model_save_path, 'checkpoint_{}.pth'.format(epoch)))
            logger.info('Saved model at epoch {}'.format(epoch))
            # Remove old checkpoint
            if epoch > 0:
                old_checkpoint = os.path.join(model_save_path, 'checkpoint_{}.pth'.format(epoch - 1))
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info('Removed old checkpoint {}'.format(old_checkpoint))
    
    logger.info('End training teacher model')
    sys.exit(0)


if "self_kd_config" in config:
    temperature = float(config['self_kd_config']['temperature'])
    alpha = float(config['self_kd_config']['alpha'])
    beta = float(config['self_kd_config']['beta'])

if 'criterions' in config and 'criterion_weights' in config:
    logger.info('Use mid level loss')
    logger.info('Mid level loss config: {}'.format(config['criterions']))
    logger.info('Mid level loss weight: {}'.format(config['criterion_weights']))

for epoch in tqdm(range(num_epochs), colour='green'):
    logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

    ## Freeze SSL model for the first defined epochs
    if 'freeze_ssl_num_epoch' in config['train'] and epoch < config['train']['freeze_ssl_num_epoch']:
        logger.info('Freeze SSL model')
        student_model.module.ssl_model.frozen()
        
    else:
        if student_model.module.ssl_model.freeze:
            logger.info('Unfreeze SSL model')
            student_model.module.ssl_model.unfrozen()
        else:
            ## Do nothing
            pass

        
    if "self_kd_config" not in config:
        
        train_loss = kd_train_epoch(train_loader, student_model, teacher_model, optimizer, device, scaler, config, exp_lr_scheduler, use_amp = use_amp)
        eval_loss, accuracy = kd_val_epoch(dev_loader, student_model, device)
        logger.info('Epoch: {} - train_loss: {} - eval_loss: {}'.format(epoch, train_loss, eval_loss))
    else:
        train_loss, train_total_label_loss, train_total_kd_loss, train_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss = self_KD_teacher_train_epoch(train_loader, student_model, teacher_model, optimizer, device, scaler, config, exp_lr_scheduler, temperature=temperature, alpha=alpha, beta=beta, use_amp = use_amp)
        # Eval
        eval_loss, accuracy = self_KD_teacher_val_epoch(dev_loader, student_model, device)
        writer.add_scalar('Loss/train_label', train_total_label_loss, epoch)
        writer.add_scalar('Loss/train_kd', train_total_kd_loss, epoch)
        writer.add_scalar('Loss/train_feature', train_total_feature_loss, epoch)
        writer.add_scalar('Loss/train_hidden_rep', running_total_hidden_rep_loss, epoch)
        writer.add_scalar('Loss/train_sup_contrastive', running_sup_contrastive_loss, epoch)
        logging.log(logging.INFO, 'Epoch: {} - train_loss: {} - train_total_label_loss: {} - train_total_kd_loss: {} - train_total_feature_loss: {} - running_total_hidden_rep_loss: {} - train_sup_contrastive: {} - eval_loss: {}'.format(epoch, train_loss, train_total_label_loss, train_total_kd_loss, train_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss, eval_loss))

    if exp_lr_scheduler is not None:
        if config['learning_rate_scheduler']['name'] == 'ReduceLROnPlateau':
            exp_lr_scheduler.step(eval_loss)
        elif config['learning_rate_scheduler']['name'] == 'MultiStepLR' or config['learning_rate_scheduler']['name'] == 'StepLR':
            exp_lr_scheduler.step()
    

    # Log
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/eval', eval_loss, epoch)
    writer.add_scalar('Accuracy/eval', accuracy, epoch)
    # Write current learning rate to tensorboard

    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        writer.add_scalar('Lr/epoch', exp_lr_scheduler.get_last_lr()[0], epoch)
        wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "learning_rate": exp_lr_scheduler.get_last_lr()[0]})
    else:
        writer.add_scalar('Lr/epoch', optimizer.param_groups[0]['lr'], epoch)
        wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "learning_rate": optimizer.param_groups[0]['lr']})

    if train_loss < 0.001 and eval_loss < 0.001:
        wandb.alert(
            title='Low loss',
            text=f'train_loss {train_loss} and eval_loss: {eval_loss} is below the acceptable threshold 0.001',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=5)
        )
    # Early stopping
    early_stopping(eval_loss, student_model, epoch)
    if early_stopping.early_stop:
        logger.info("Early stopping")
        break
    # Save model
    if epoch % 1 == 0:
        torch.save(student_model.state_dict(), os.path.join(model_save_path, 'checkpoint_{}.pth'.format(epoch)))
        logger.info('Saved model at epoch {}'.format(epoch))
        # Remove old checkpoint
        if epoch > 0:
            old_checkpoint = os.path.join(model_save_path, 'checkpoint_{}.pth'.format(epoch - 1))
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.info('Removed old checkpoint {}'.format(old_checkpoint))


if use_amp:
    logger.info('End automatic mixed precision training')
