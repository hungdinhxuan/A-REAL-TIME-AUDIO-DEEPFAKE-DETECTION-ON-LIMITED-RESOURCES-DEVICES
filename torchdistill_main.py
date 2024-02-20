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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Device: {}'.format(device))

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
    if config["model"]["teacher"]["pretrained_path"] == "":
        logger.info("No pretrained teacher path")
    else:
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
        
        train_loss = kd_train_epoch(train_loader, student_model, teacher_model, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager, exp_lr_scheduler, use_amp = use_amp)
        eval_loss, accuracy = kd_val_epoch(dev_loader, student_model, device, config)
        logger.info('Epoch: {} - train_loss: {} - eval_loss: {}'.format(epoch, train_loss, eval_loss))
    else:
        train_loss, train_total_label_loss, train_total_kd_loss, train_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss = self_KD_teacher_train_epoch(train_loader, student_model, teacher_model, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager, exp_lr_scheduler, temperature=temperature, alpha=alpha, beta=beta, use_amp = use_amp)
        # Eval
        eval_loss, accuracy = self_KD_teacher_val_epoch(dev_loader, student_model, device, config)
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
