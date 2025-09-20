import torch
import os
import sys
from wav2vec2_vib import Model as Wav2Vec2VIB

from student import *
from teacher import *
from data_utils import *
from torchdistill_utils_alpha_test_v3 import *
from utils import *
from torchdistill.models.registry import get_model

from torchdistill.core.forward_hook import ForwardHookManager
from losses import *
import yaml
from startup_config import set_random_seed
from menu import get_main_menu
from main import get_train_dev_dataloader


import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
from main import W2V2_TA
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import wandb
from datetime import timedelta
from wandb import AlertLevel
from contrast.supcontrastloss import SupConLoss
from engine.dot import DistillationOrientedTrainer
import eval_metrics_DF as em
from aasist.AASIST import *
from wav2vec2_conformertcm import Model as W2V2_ConformerTCM
import numpy as np
from losses import StandardMidLoss_v4 as StandardMidLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disable pydub logging
logging.getLogger('pydub.converter').setLevel(logging.CRITICAL)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Device: {}'.format(device))

args = get_main_menu()
# Load configuration
with open(args.yaml, 'r') as f:
    logger.info('Load configuration file {}'.format(args.yaml))
    config = yaml.safe_load(f)


seed = config['train'].get('seed', 1234)
ce_weight = torch.FloatTensor(config['train'].get(
    'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
student_resume_path = config['train'].get('student_resume', None)
sup_contrastive = config['train'].get('sup_contrastive', False)
is_learning_rate_scheduler = config['train'].get(
    'is_learning_rate_scheduler', False)
patience = config['train'].get('patience', 10)
use_amp = config['train'].get('amp', False)
learning_rate_scheduler_name = config['learning_rate_scheduler'].get(
    'name', None)
student_model_name = config['model']['student'].get(
    'name', 'Distil_W2V2BASE_AASISTL')
teacher_model_name = config['model']['teacher'].get('name', 'W2V2_TA')
model_path = config['model']['teacher'].get('pretrained_path', None)
student_model_path = config['train'].get('student_resume', None)
student_model_type = config['model']['student'].get(
    'type', 'ssl')
augment_mode = config["train"].get("augment_mode", "rawboost")
dataset = config["train"].get("dataset", "LA19")
dot = config["train"].get("dot", False)
mixup = config["train"].get("mixup", False)
restore = config["train"].get("restore", False)
teacher_dict = config["model"].get("teacher_multi", {})
copy_weights = config["train"].get("copy_weights", False)
padding_size = config["train"].get(
    "padding_size", 64600)  # default 64600 for 4s

# Model saving and averaging configuration
n_mejores = config['train'].get('n_mejores_loss', 5)  # number of best models to save
average_model = config['train'].get('average_model', True)  # whether to average models
n_average_model = config['train'].get('n_average_model', 5)  # number of models to average

logger.info("Padding size: {}".format(padding_size))
logger.info("Number of best models to save: {}".format(n_mejores))
logger.info("Average models: {}, Number to average: {}".format(average_model, n_average_model))

is_teacher_parallel = config["model"]["teacher"].get("is_parallel", True)
freeze_layers = config["train"].get("freeze_layers", [])
encoder_layerdrop = config["model"]["student"].get("kwargs", {}).get(
    "encoder_layerdrop", 0.0)


torchaudio_wrapper = config["train"].get("torchaudio_wrapper", False)


custom_order_copy_weights = config["model"]["student"].get("kwargs", {}).get(
    "custom_order", [])
order = config["model"]["student"].get("kwargs", {}).get(
    "order", None)

wandb_project_name = config["train"].get("wandb_project_name", "torchdistill_v2")

ssl_teacher_path = os.getenv("XLSR_PRETRAINED_PATH")
ssl_student_path = os.getenv("XLSR_PRETRAINED_PATH")

if augment_mode == "rawboost":
    # DEFAULT rawboost 3
    args.algo = config["train"].get("algo", 3)

# Wandb
wandb_disabled = config["train"].get("wandb_disabled", False)
args.batch_size = config['train'].get('batch_size', 32)

if not wandb_disabled:
    wandb.init(project=wandb_project_name, config={
        **config
    }, name=config['name'])

set_random_seed(seed, args)
logger.info('Random seed: {}'.format(seed))


teacher_model = get_model(config['model']['teacher']['name'],
                          device=device, ssl_cpkt_path=ssl_teacher_path, **config['model']['teacher']['kwargs']).to(device)

student_model = get_model(
            student_model_name, device=device, **config['model']['student']['kwargs']).to(device)

print("Current number of student parameters: ",  sum(p.numel()
      for p in student_model.parameters()))

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
    if config["model"]["teacher"]["pretrained_path"] == "":
        logger.info("No pretrained teacher path")
    else:
        try:
            teacher_model.load_state_dict(torch.load(
                config["model"]["teacher"]["pretrained_path"], map_location=device))
            logger.info("Loaded teacher model from {}".format(
                config["model"]["teacher"]["pretrained_path"]))

        except Exception as e:
            logger.info("Failed to load teacher model from {}".format(
                config["model"]["teacher"]["pretrained_path"]))
            print(e)
            sys.exit(0)
else:
    teacher_model.load_state_dict(torch.load(
        args.model_path, map_location=device))
    logger.info("Loaded teacher model from {}".format(args.model_path))

# DEBUG
# sys.exit(0)

if "student_resume" in config["train"] and config["train"]["student_resume"] != "":
    student_model.load_state_dict(torch.load(
        config["train"]["student_resume"], map_location=device), strict=False)
    logger.info("Loaded student model from {}".format(
        config["train"]["student_resume"]))


if copy_weights:
    if is_teacher_parallel:
        student_model.module.load_state_dict(
            teacher_model.module.state_dict(), strict=False)
    else:
        student_model.module.load_state_dict(
            teacher_model.state_dict(), strict=False)

    logger.info("Copied teacher weights to student")

    if len(custom_order_copy_weights) > 0 and order == "custom":
        logger.info("Copy transformer weights with custom order")
        for index, value in enumerate(custom_order_copy_weights):
            if is_teacher_parallel:
                student_model.module.front_end.model.encoder.layers[index].load_state_dict(
                    teacher_model.module.front_end.model.encoder.layers[value].state_dict(), strict=False)
            else:
                student_model.module.front_end.model.encoder.layers[index].load_state_dict(
                    teacher_model.front_end.model.encoder.layers[value].state_dict(), strict=False)
            logger.info(
                f"Copied teacher transformer weights from  to front_end.model.encoder.layers[{value}] to front_end.model.encoder.layers[{index}] student")

if torchaudio_wrapper:
    logger.info('Use torchaudio wrapper')
    if is_teacher_parallel:
        teacher_model = teacher_model.module
        is_teacher_parallel = False

    teacher_model.front_end = W2V2_TA(
        import_fairseq_model(teacher_model.front_end.model)).to(device)

    print(teacher_model)


# Register forward hook
logger.info('Register forward hook for teacher')
for module_path, ios in zip(config['model']['teacher']['teacher_module_paths'], config['model']['teacher']['teacher_module_ios']):
    logger.info('Register teacher forward hook for {}'.format(module_path))
    requires_input, requires_output = ios.split(':')
    requires_input, requires_output = bool(
        requires_input), bool(requires_output)
    if "is_parallel" in config["model"]["teacher"] and not is_teacher_parallel:
        teacher_forward_hook_manager.add_hook(
            teacher_model, module_path, requires_input=requires_input, requires_output=requires_output)
    else:
        teacher_forward_hook_manager.add_hook(
            teacher_model.module, module_path, requires_input=requires_input, requires_output=requires_output)


logger.info('Register forward hook for student')
for module_path, ios in zip(config['model']['student']['student_module_paths'], config['model']['student']['student_module_ios']):
    logger.info('Register student forward hook for {}'.format(module_path))
    requires_input, requires_output = ios.split(':')
    requires_input, requires_output = bool(
        requires_input), bool(requires_output)
    student_forward_hook_manager.add_hook(
        student_model.module, module_path, requires_input=requires_input, requires_output=requires_output)


logger.info('Prepare training, dev set .....')

logger.info(f'Use {augment_mode} data augmentation')
if dataset:
    logger.info(f'Use {dataset} dataset')

train_loader, dev_loader = get_train_dev_dataloader(
    args, augment_mode, dataset, padding_size=padding_size)


standard_mid_loss = StandardMidLoss(t_layers=len(config['model']['teacher']['teacher_module_paths'])-1, s_layers=len(config['model']['student']['student_module_paths'])-1, device=device)

# Add standard_mid_loss parameters to optimizer
optimizer = torch.optim.Adam([{
    'params': student_model.parameters(),
},
{
    'params': standard_mid_loss.parameters(),
}
], lr=float(
    config['train']['learning_rate']), weight_decay=config['train']['weight_decay'])

def check_optimizer_includes_loss_params(optimizer, loss_module):
    """Check if optimizer includes loss module parameters"""
    
    # Get all loss module parameters
    loss_param_ids = {id(p) for p in loss_module.parameters()}
    
    # Get all optimizer parameters
    optimizer_param_ids = set()
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            optimizer_param_ids.add(id(param))
    
    # Check coverage
    missing_params = loss_param_ids - optimizer_param_ids
    
    print("=== Optimizer Parameter Check ===")
    print(f"Loss module has {len(loss_param_ids)} parameters")
    print(f"Optimizer covers {len(optimizer_param_ids & loss_param_ids)} loss parameters")
    print(f"Missing {len(missing_params)} loss parameters from optimizer")
    
    if missing_params:
        print("\nMISSING PARAMETERS:")
        for name, param in loss_module.named_parameters():
            if id(param) in missing_params:
                print(f"  - {name}: {param.shape}")
                print(f"    requires_grad: {param.requires_grad}")
        return False
    else:
        print("✅ All loss module parameters are in optimizer!")
        return True

check_optimizer_includes_loss_params(optimizer, standard_mid_loss)

# sys.exit(0)

exp_lr_scheduler = None
if 'is_learning_rate_scheduler' in config and config['is_learning_rate_scheduler']:

    # Initialize learning rate scheduler by using its name and its parameters
    exp_lr_scheduler = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler']['name'])(
        optimizer, **config['learning_rate_scheduler']['params'])
    logger.info(
        f'Use learning rate scheduler {config["learning_rate_scheduler"]["name"]}, {exp_lr_scheduler}')
else:
    logger.info('No learning rate scheduler')


scaler = torch.cuda.amp.GradScaler(enabled=config['train']['amp'])
writer = SummaryWriter('logs/{}'.format(config['name']))
# model_save_path = os.path.join("models", config['name'])

# folder to saved
model_to_save = config['train'].get('model_to_save', 'runs')

if not os.path.exists(model_to_save):
    os.makedirs(model_to_save, exist_ok=True)

model_save_path = os.path.join(model_to_save, config['name'])  # Change to runs

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    logger.info('Created model save path {}'.format(model_save_path))

# Create best models directory for top-n checkpoint saving
best_save_path = os.path.join(model_save_path, 'best')
if not os.path.exists(best_save_path):
    os.makedirs(best_save_path)
    logger.info('Created best models save path {}'.format(best_save_path))




# Train loop
logger.info("Start training")
num_epochs = config['train']['num_epochs']



if 'criterions' in config and 'criterion_weights' in config:
    logger.info('Use mid level loss')
    logger.info('Mid level loss config: {}'.format(config['criterions']))
    logger.info('Mid level loss weight: {}'.format(
        config['criterion_weights']))

start_epoch = 0

# Initialize early stopping variables
not_improving = 0
best_loss = float('inf')
bests = np.ones(n_mejores, dtype=float) * float('inf')

# Initialize best checkpoints tracking (no placeholder files needed)

if restore:
    logger.info('Restore from previous checkpoint')
    previous_model_saved_path = os.path.join(model_to_save, config['name'])

    if not os.path.exists(previous_model_saved_path):
        logger.info(
            'Previous model saved path {} does not exist'.format(previous_model_saved_path))
        sys.exit(0)

    # Get the latest checkpoint startwith 'checkpoint'
    checkpoints = [f for f in os.listdir(
        previous_model_saved_path) if f.startswith('checkpoint')]
    if len(checkpoints) == 0:
        logger.info('No checkpoint found')
        sys.exit(0)
    # Sort the checkpoint by epoch
    checkpoints = sorted(checkpoints, key=lambda x: int(
        x.split('_')[-1].split('.')[0]))
    last_checkpoint = checkpoints[-1]

    student_model_path = os.path.join(
        previous_model_saved_path, last_checkpoint)
    checkpoint = torch.load(student_model_path)
    try:

        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        scaler.load_state_dict(checkpoint['scaler'])
        logger.info(
            'Restore from previous checkpoint at epoch {}'.format(start_epoch))

    except:
        logger.info(
            'Failed to restore from previous checkpoint, the checkpoint may be corrupted or deprecated')
        sys.exit(0)

######## Transformer layer drop ########
if encoder_layerdrop > 0:
    logger.info('Use encoder layer drop')
    student_model.module.front_end.model.cfg.encoder_layerdrop = encoder_layerdrop
    logger.info('Encoder layer drop: {}'.format(encoder_layerdrop))


######### Freeze layers #########
if len(freeze_layers) > 0:
    logger.info('Freeze layers')
    for name, param in student_model.module.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False
            logger.info(f'Freeze {name}')


# Summary model
summary(student_model, (1, 16000))

for epoch in tqdm(range(start_epoch, num_epochs), colour='green'):
    logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

    # Freeze SSL model for the first defined epochs
    if 'freeze_ssl_num_epoch' in config['train'] and epoch < config['train']['freeze_ssl_num_epoch']:
        logger.info('Freeze SSL model')
        student_model.module.front_end.frozen()

    else:
        try:
            if student_model.module.front_end.freeze:
                logger.info('Unfreeze SSL model')
                student_model.module.front_end.unfrozen()
            else:
                # Do nothing
                pass
        except:
            pass

    if "self_kd_config" not in config:
        train_loss, train_acc, loss_dict = kd_train_epoch(train_loader, student_model, teacher_model, optimizer, device, scaler, config,
                                                              student_forward_hook_manager, teacher_forward_hook_manager, epoch, exp_lr_scheduler=exp_lr_scheduler, use_amp=use_amp, standard_mid_loss=standard_mid_loss)
       
        # dev_loss, accuracy = kd_val_epoch(
        #     dev_loader, student_model, device, config)
        # new eval
        dev_loss, accuracy, dev_loss_dict = kd_val_epoch_advanced(
            dev_loader, student_model, teacher_model, device, config, student_forward_hook_manager, teacher_forward_hook_manager, standard_mid_loss=standard_mid_loss)

        logger.info(
            'Epoch: {} - train_loss: {} - dev_loss: {}'.format(epoch, train_loss, dev_loss))
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        for key, value in loss_dict.items():
            if isinstance(value, AverageMeter):
                wandb.log({key: value.avg}) 
            else:
                wandb.log({key: value})
            # writer.add_scalar(f'train_key', value, epoch)

        for key, value in dev_loss_dict.items():
            if isinstance(value, AverageMeter):
                wandb.log({"Eval/" + key: value.avg})
            else:
                wandb.log({"Eval/" + key: value})
            # writer.add_scalar(f'eval_key', value, epoch)

        wandb.log({
            "Accuracy_train": train_acc
        })

    if exp_lr_scheduler is not None:
        if config['learning_rate_scheduler']['name'] == 'ReduceLROnPlateau':
            exp_lr_scheduler.step(dev_loss)
        elif config['learning_rate_scheduler']['name'] == 'MultiStepLR' or config['learning_rate_scheduler']['name'] == 'StepLR':
            exp_lr_scheduler.step()


    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        writer.add_scalar('Lr/epoch', exp_lr_scheduler.get_last_lr()[0], epoch)

        
        wandb.log({"train_loss": train_loss, "dev_loss": dev_loss, "Eval Accuracy": accuracy,
                       "learning_rate": exp_lr_scheduler.get_last_lr()[0]})
        
    else:
        writer.add_scalar('Lr/epoch', optimizer.param_groups[0]['lr'], epoch)
        wandb.log({"train_loss": train_loss, "dev_loss": dev_loss,
                       "Eval Accuracy": accuracy,
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       "Accuracy_train": train_acc
                       })


    # Early stopping based on not_improving criteria
    current_dev_loss = dev_loss.avg if isinstance(dev_loss, AverageMeter) else dev_loss
    if current_dev_loss < best_loss:
        best_loss = current_dev_loss
        torch.save(student_model.state_dict(), os.path.join(
            model_save_path, 'best.pth'))
        logger.info('New best epoch with dev_loss: {}'.format(current_dev_loss))
        not_improving = 0
    else:
        not_improving += 1
        logger.info('Not improving for {} epochs'.format(not_improving))
    
    # Save top n_mejores models
    for i in range(n_mejores):
        if bests[i] > current_dev_loss:
            # Shift worse models down
            for t in range(n_mejores-1, i, -1):
                bests[t] = bests[t-1]
                old_path = os.path.join(best_save_path, 'best_{}.pth'.format(t-1))
                new_path = os.path.join(best_save_path, 'best_{}.pth'.format(t))
                if os.path.exists(old_path):
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    os.rename(old_path, new_path)
            
            # Save current model at position i
            bests[i] = current_dev_loss
            torch.save(student_model.state_dict(), os.path.join(
                best_save_path, 'best_{}.pth'.format(i)))
            logger.info('Saved model at rank {} with dev_loss: {}'.format(i, current_dev_loss))
            break
    
    logger.info('Current n-best losses: {}'.format(bests))
    
    # Check if we should stop early
    if not_improving >= patience:
        logger.info("Early stopping: not improving for {} epochs".format(patience))
        break



if use_amp:
    logger.info('End automatic mixed precision training')

# Model averaging after training
logger.info('######## Post-training Model Averaging ########')
if average_model and n_average_model <= n_mejores:
    logger.info('Averaging top {} models'.format(n_average_model))
    
    # Load first model
    first_model_path = os.path.join(best_save_path, 'best_0.pth')
    if os.path.exists(first_model_path):
        student_model.load_state_dict(torch.load(first_model_path, map_location=device))
        logger.info('Model loaded: {}'.format(first_model_path))
        sd = student_model.state_dict()
        
        # Add remaining models
        for i in range(1, n_average_model):
            model_path = os.path.join(best_save_path, 'best_{}.pth'.format(i))
            if os.path.exists(model_path):
                student_model.load_state_dict(torch.load(model_path, map_location=device))
                logger.info('Model loaded: {}'.format(model_path))
                sd2 = student_model.state_dict()
                for key in sd:
                    sd[key] = sd[key] + sd2[key]
            else:
                logger.warning('Model {} not found, skipping'.format(model_path))
        
        # Average the weights
        for key in sd:
            sd[key] = sd[key] / n_average_model
        
        # Load averaged model and save
        student_model.load_state_dict(sd)
        averaged_model_path = os.path.join(best_save_path, 'avg_{}_best.pth'.format(n_average_model))
        torch.save(student_model.state_dict(), averaged_model_path)
        
        logger.info('Model loaded average of {} best models and saved to {}'.format(
            n_average_model, averaged_model_path))
    else:
        logger.warning('No best models found for averaging')
else:
    if not average_model:
        logger.info('Model averaging disabled')
    else:
        logger.warning('Cannot average {} models when only {} best models are saved'.format(
            n_average_model, n_mejores))
    
    # Load single best model
    best_model_path = os.path.join(model_save_path, 'best.pth')
    if os.path.exists(best_model_path):
        student_model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info('Loaded single best model: {}'.format(best_model_path))
    else:
        logger.warning('No best model found at {}'.format(best_model_path))
