import torch
from torchdistill.models.registry import get_model
from torchdistill.core.forward_hook import ForwardHookManager
from data_utils import *
from student import *
from self_kd_zoo import *
import yaml
from startup_config import set_random_seed
from menu import get_main_menu
from main import get_train_dev_dataloader
from utils import EarlyStopping
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm
from main import W2V2_TA
from contrast.supcontrastloss import SupConLoss
import wandb

class DistillKL(nn.Module):
    """
    A class used to represent the DistillKL module
    """
    def __init__(self, temperature):
        """
        Create a new DistillKL object
        :param temperature: a temperature parameter
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, y_s, y_t):
        """
        Forward pass for the DistillKL module
        :param y_s: student outputs
        :param y_t: teacher outputs
        :return: the KL divergence between the student and teacher outputs
        """
        p_s = F.log_softmax(y_s/self.temperature, dim=1)
        p_t = F.softmax(y_t/self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        return loss

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
            # elif method == 'virtual_softmax':
            #     logit = student(batch_x, batch_y, loss_type='virtual_softmax')
            #     loss_cls += criterion_cls(logit, batch_y)
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Device: {}'.format(device))


args = get_main_menu()
# Load configuration
with open(args.yaml, 'r') as f:
    logger.info('Load configuration file {}'.format(args.yaml))
    config = yaml.safe_load(f)


seed = config['train']['seed']

set_random_seed(seed, args)
logger.info('Random seed: {}'.format(seed))

# Get variables from config
learning_rate = float(config['train']['learning_rate'])
weight_decay = float(config['train']['weight_decay'])
num_epochs = config['train']['num_epochs']
augment_mode = config["train"].get("augment_mode", "rawboost")
dataset = config["train"].get("dataset", "LA19")
method = config["train"].get("method", "cross_entropy")
T = config["train"].get("T", 1.0)
ce_weight = torch.FloatTensor(config['train'].get('cross_entropy_loss_weight', [0.1, 0.9])).to(device)
student_resume_path = config['train'].get('student_resume', None)
sup_contrastive = config['train'].get('sup_contrastive', False)
is_learning_rate_scheduler = config.get('is_learning_rate_scheduler', False)
patience = config['train'].get('patience', 10)
use_amp = config['train'].get('amp', False)
learning_rate_scheduler_name = config['learning_rate_scheduler'].get('name', None)
student_model_name = config['model']['student'].get('name', 'Distil_W2V2BASE_AASISTL')

## Wandb
wandb.init(project="selfdistill", config={
    **config,
},
name=config['name']
)

args.batch_size = config['train']['batch_size']

student_model = get_model(student_model_name, device=device).to(device)

# student_forward_hook_manager = ForwardHookManager(device)
student_model = torch.nn.DataParallel(student_model).to(device)

if student_resume_path:
    student_resume_strict = config["train"].get("student_resume_strict", False)
    state_dict = torch.load(student_resume_path, map_location=device)
    student_model.load_state_dict(state_dict, strict=student_resume_strict)
    logger.info("Loaded student model from {} with mode {}".format(student_resume_path, "strict" if student_resume_strict else "non-strict"))


logger.info('Prepare training, dev set .....')
logger.info(f'Use {augment_mode} data augmentation')
logger.info(f'Use {dataset} dataset')
logger.info(f'Use {method} self-kd method')

if sup_contrastive:
    logger.info('Use supervised contrastive learning')
    train_loader, dev_loader = get_train_dev_dataloader_contrastive(args)
else:
    train_loader, dev_loader = get_train_dev_dataloader(args, augment_mode, dataset)

optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate,weight_decay=weight_decay)

exp_lr_scheduler = None

if is_learning_rate_scheduler:
    logger.info(f'Use learning rate scheduler {config["learning_rate_scheduler"]["name"]}')
    # Initialize learning rate scheduler by using its name and its parameters
    exp_lr_scheduler = getattr(torch.optim.lr_scheduler, learning_rate_scheduler_name)(optimizer, **config['learning_rate_scheduler']['params'])
else:
    logger.info('Do not use learning rate scheduler')


scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
writer = SummaryWriter('logs/{}'.format(config['name']))
model_save_path = os.path.join("models", config['name'])

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    logger.info('Created model save path {}'.format(model_save_path))


early_stopping = EarlyStopping(patience=patience, verbose=True, model_save_path=model_save_path)

# Train loop
logger.info("Start training")


criterion_cls = nn.CrossEntropyLoss(ce_weight)
criterion_div = DistillKL(T)
criterion_list = nn.ModuleList([])

criterion_list.append(criterion_cls)  # classification loss
criterion_list.append(criterion_div)
criterion_list.to(device)


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

    # Train
    train_loss, train_loss_cls, train_loss_div = kd_train_epoch(train_loader, student_model, optimizer, device, scaler, config, criterion_list, method, exp_lr_scheduler, use_amp)
    # Eval
    eval_loss, accuracy = kd_val_epoch(dev_loader, student_model, device, criterion_cls)

    logger.info('Epoch {}/{}: train_loss: {:.4f}, train_loss_cls: {:.4f}, train_loss_div: {:.4f}, eval_loss: {:.4f}, accuracy: {:.2f}'.format(epoch, num_epochs - 1, train_loss, train_loss_cls, train_loss_div, eval_loss, accuracy))

    if exp_lr_scheduler is not None:
        if learning_rate_scheduler_name == 'ReduceLROnPlateau':
            exp_lr_scheduler.step(eval_loss)
        elif learning_rate_scheduler_name == 'MultiStepLR':
            exp_lr_scheduler.step()
    

    # Log
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/train_cls', train_loss_cls, epoch)
    writer.add_scalar('Loss/train_div', train_loss_div, epoch)
    writer.add_scalar('Loss/eval', eval_loss, epoch)
    writer.add_scalar('Accuracy/eval', accuracy, epoch)
    # Write current learning rate to tensorboard

    if exp_lr_scheduler is not None and learning_rate_scheduler_name != 'ReduceLROnPlateau':
        writer.add_scalar('Lr/epoch', exp_lr_scheduler.get_last_lr()[0], epoch)
        wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "learning_rate": exp_lr_scheduler.get_last_lr()[0]})
    else:
        writer.add_scalar('Lr/epoch', optimizer.param_groups[0]['lr'], epoch)
        wandb.log({"train_loss": train_loss, "eval_loss": eval_loss, "learning_rate": optimizer.param_groups[0]['lr']})

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
