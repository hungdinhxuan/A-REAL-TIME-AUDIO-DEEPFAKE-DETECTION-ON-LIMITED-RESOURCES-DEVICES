import torch
from torchdistill.models.registry import get_model
from torchdistill.losses.registry import get_mid_level_loss
from torchdistill.core.forward_hook import ForwardHookManager
from data_utils import *
from student import *
from teacher import *
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
# from neural_compressor.training import prepare_compression
from transformers  import  AutoConfig, WavLMModel
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Device: {}'.format(device))

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

def self_KD_teacher_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, exp_lr_scheduler=None, temperature: float =3, alpha: float =0.1, beta: float = 1e-6,  use_amp: bool = True):
    logger.info('Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {}'.format(temperature, alpha, beta))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    running_total_hidden_rep_loss = 0
    
    
    student.train()
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

def kd_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, exp_lr_scheduler=None,  use_amp: bool = True):
    logger.info('Training KD')
    running_loss = 0
    
    student.train()
    teacher.eval()
    weight_config = config['train']['cross_entropy_loss_weight']

    if "alpha" not in config['train']:
        forward_target = True
    else:
        forward_target = False
        alpha = float(config['train']['alpha'])

        if 'beta' in config['train']:
            beta = float(config['train']['beta'])
        else:
            beta = 0.5
  
    weight = torch.FloatTensor(weight_config).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    num_total = 0.0
    
    if not config['train']['teacher']:
        logger.info('No teacher')
        del teacher
    
    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        logger.info("Current learning rate: {}".format(exp_lr_scheduler.get_last_lr()[0]))
    else:
        logger.info("Current learning rate: {}".format(optimizer.param_groups[0]['lr']))
        
    iters = len(train_loader)
    # Create a progress bar
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
                    t_logits = teacher(batch_x)
                    teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

            batch_y = batch_y.view(-1).type(torch.int64).to(device)


            # Multiple loss
            total_loss = 0

            # Check if key exists

            if 'criterions' in config and 'criterion_weights' in config:

                if len(config['criterions']) != len(config['criterion_weights']):
                    raise ValueError('Number of criterions and criterion_weights must be the same')

                for loss, weight in zip(config['criterions'], config['criterion_weights']):
                    weight = float(weight)
                    loss_i = get_mid_level_loss(mid_level_criterion_config = loss)

                    if forward_target:
                        total_loss += (loss_i.forward(student_io_dict, teacher_io_dict, batch_y) * weight)
                    else:
                        total_loss += (loss_i.forward(student_io_dict, teacher_io_dict) * weight)

            if not forward_target:
                batch_loss = criterion(batch_out, batch_y)
                total_loss +=  (alpha * batch_loss)

                ## Total loss + KL divergence loss
                
                total_loss += (beta * F.kl_div(F.log_softmax(batch_out, dim=1), F.softmax(t_logits, dim=1), reduction='batchmean'))

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
        

    running_loss /= num_total
    
    return running_loss

def kd_val_epoch(dev_loader, model, device):
    logger.info('Validation ----')
    val_loss = 0
    model.eval()
    weight_config = config['train']['cross_entropy_loss_weight']
    
    weight = torch.FloatTensor(weight_config).to(device)
    
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
        
            val_loss += (batch_loss.item() * batch_size)

            probabilities = F.softmax(batch_out, dim=1)
            predicted_labels = (probabilities[:,0] >= 0.5).int()

            num_correct += (predicted_labels == batch_y).sum().item()
            

        accuracy = (num_correct / num_total) * 100
        print("accuracy",accuracy)
        val_loss /= num_total
        print('[VALIDATION] eval_accuracy: ', accuracy)
        return val_loss, accuracy

args = get_main_menu()
# Load configuration
with open(args.yaml, 'r') as f:
    logger.info('Load configuration file {}'.format(args.yaml))
    config = yaml.safe_load(f)

seed = config['train']['seed']
set_random_seed(seed, args)
logger.info('Random seed: {}'.format(seed))


teacher_model = get_model(config['model']['teacher']['name'], device=device).to(device)
student_model = get_model(config['model']['student']['name'], device=device).to(device)

teacher_forward_hook_manager = ForwardHookManager(device)
# student_forward_hook_manager = ForwardHookManager(device)

#student_model = torch.nn.DataParallel(student_model).to(device)

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


# if "student_resume" in config["train"]:
#     student_model.load_state_dict(torch.load(config["train"]["student_resume"],map_location=device))
#     logger.info("Loaded student model from {}".format(config["train"]["student_resume"]))

# huggingface_url='microsoft/wavlm-base-plus'
# # torch modules
# teacher = WavLMModel.from_pretrained(
#    huggingface_url,
#     config=AutoConfig.from_pretrained(huggingface_url),
#     ignore_mismatched_sizes=False,
# )

# for idx, (name, layer) in enumerate(teacher.named_modules()):
#     print(f"Index: {idx}, Layer Name: {name}, Layer Type: {layer.__class__.__name__}")