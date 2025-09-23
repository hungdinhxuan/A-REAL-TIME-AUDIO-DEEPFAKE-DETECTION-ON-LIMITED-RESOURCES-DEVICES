import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torchdistill.losses.registry import get_mid_level_loss
from contrast.supcontrastloss import SupConLoss, supcon_loss
import numpy as np
from utils import AverageMeter
from unified_loss import UnifiedMidLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('pydub.converter').setLevel(logging.CRITICAL)
logging.getLogger('hydra.core.utils').setLevel(logging.CRITICAL)

def create_unified_loss_from_config(config: dict, t_layers: int, s_layers: int, device: torch.device):
    """
    Create UnifiedMidLoss from YAML configuration
    
    Args:
        config: YAML configuration dictionary
        t_layers: Number of teacher layers
        s_layers: Number of student layers
        device: Device to use
    
    Returns:
        UnifiedMidLoss instance configured according to config
    """
    print("Creating unified loss from config ", config)
    
    unified_loss_config = config.get('unified_loss', {})
    
    #print(config.keys())
    
    # Get projection configuration
    projection_config = unified_loss_config.get('projection', {})
    if not projection_config:
        projection_config = {
            'type': None,
            'target': 'teacher',
            'input_dim': 1024,
            'output_dim': 768,
            'kwargs': {}
        }
        # WARNING: No projection configuration provided
        logger.warning("No projection configuration provided")
    
    # Get loss configuration
    loss_config = unified_loss_config.get('loss', {})
    if not loss_config:
        loss_config = {
            'enabled': ['mse', 'cosine'],
            'weights': {'mse': 0.0001, 'cosine': 1.0, 'recon': 0.0001, 'l1': 0.0001}
        }
        # WARNING: No loss configuration provided
        logger.warning("No loss configuration provided")
    # Get pooling configuration
    pooling_config = unified_loss_config.get('pooling', {})
    if not pooling_config:
        pooling_config = {'method': 'mean'}
        # WARNING: No pooling configuration provided
        logger.warning("No pooling configuration provided")
    # Get processing mode
    processing_mode = unified_loss_config.get('processing_mode', 'legacy')
    if not processing_mode:
        processing_mode = 'legacy'
        # WARNING: No processing mode provided
        logger.warning("No processing mode provided")
    
    return UnifiedMidLoss(
        t_layers=t_layers,
        s_layers=s_layers,
        device=device,
        projection_config=projection_config,
        loss_config=loss_config,
        pooling_config=pooling_config,
        processing_mode=processing_mode,
        
        # **unified_loss_config except for the above keys
        **{k: v for k, v in unified_loss_config.items() if k not in ['projection', 'loss', 'pooling', 'processing_mode']}
    )


def kd_train_epoch_unified(train_loader, student, teacher, optimizer, device, scaler, config, 
                          student_forward_hook_manager, teacher_forward_hook_manager, epoch, 
                          exp_lr_scheduler=None, use_amp=True, unified_loss=None):
    """
    Training epoch with unified loss support
    
    Args:
        train_loader: Training data loader
        student: Student model
        teacher: Teacher model
        optimizer: Optimizer
        device: Device
        scaler: Mixed precision scaler
        config: Configuration dictionary
        student_forward_hook_manager: Student forward hook manager
        teacher_forward_hook_manager: Teacher forward hook manager
        epoch: Current epoch
        exp_lr_scheduler: Learning rate scheduler
        use_amp: Use automatic mixed precision
        unified_loss: UnifiedMidLoss instance
    """
    logger.info('Training KD with Unified Loss')
    running_loss = 0
    running_unified_loss = 0

    student.train()
    teacher.eval()

    num_correct = 0.0

    forward_target = "alpha" not in config['train']
    alpha = float(config['train'].get('alpha', 1))
    beta = float(config['train'].get('beta', 0.5))
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    is_recon_loss = config['train'].get('is_recon_loss', False)
    num_total = 0.0

    if not config['train']['teacher']:
        logger.info('No teacher')
        del teacher

    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        logger.info("Current learning rate of scheduler {}: {}".format(config['learning_rate_scheduler']['name'],
                                                                       exp_lr_scheduler.get_last_lr()[0]))
    else:
        logger.info("Current learning rate: {}".format(
            optimizer.param_groups[0]['lr']))

    iters = len(train_loader)
    # Create a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    # loss list for monitoring
    loss_dict = dict()
    loss_dict['ce_loss'] = AverageMeter()

    criterions = config.get('criterions', [])
    criterion_key_list = []

    if is_recon_loss:
        loss_dict['recon_loss'] = AverageMeter()
        loss_dict['BCE'] = AverageMeter()
        loss_dict['KLD'] = AverageMeter()

    for loss in criterions:
        student_module_path = loss.get('kwargs', {}).get(
            'student_module_path', 'default_student_module_path')
        teacher_module_path = loss.get('kwargs', {}).get(
            'teacher_module_path', 'default_teacher_module_path')
        key = loss.get('key', 'default_key')
        criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
        criterion_key_list.append(criterion_key)
        loss_dict[criterion_key] = AverageMeter()
    
    # Unified loss tracking
    if unified_loss is not None:
        loss_dict['unified_loss'] = AverageMeter()
        loss_dict['unified_loss_mse'] = AverageMeter()
        loss_dict['unified_loss_cosine'] = AverageMeter()
        unified_loss_enabled = config.get('unified_loss', {}).get('loss', {}).get('enabled', [])
        if 'recon' in unified_loss_enabled:
            loss_dict['unified_loss_recon'] = AverageMeter()
        if 'l1' in unified_loss_enabled:
            loss_dict['unified_loss_l1'] = AverageMeter()

    for i, (batch_x, batch_y) in pbar:

        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # Multiple loss
            total_loss = torch.tensor(0.).to(device)
            kd_loss = torch.tensor(0.).to(device)
            ce_loss = torch.tensor(0.).to(device)
            recon_loss = torch.tensor(0.).to(device)
            kl_loss = torch.tensor(0.).to(device)
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            if len(batch_x.shape) == 3:
                batch_x = batch_x.squeeze(0).transpose(0, 1)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            num_total += batch_size
            batch_x = batch_x.to(device)

            if config["model"]["student"]["name"].startswith("Self"):
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(
                    batch_x)
            else:
                batch_out = student(
                    batch_x)

            student_io_dict = student_forward_hook_manager.pop_io_dict()

            # Get teacher output
            if config['train']['teacher']:
                with torch.no_grad():
                    t_logits = teacher(batch_x)
                    teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

            # Check if key exists
            if 'criterions' in config and 'criterion_weights' in config:

                if len(config['criterions']) != len(config['criterion_weights']):
                    raise ValueError(
                        'Number of criterions and criterion_weights must be the same')

                for loss, weight, criterion_key in zip(config['criterions'], config['criterion_weights'], criterion_key_list):
                    weight = float(weight)

                    loss_i = get_mid_level_loss(
                        mid_level_criterion_config=loss)

                    if config['train']['teacher']:
                        tmp_loss = loss_i.forward(student_io_dict,
                                                  teacher_io_dict, size=batch_size)
                        loss_dict[criterion_key
                                  ].update(tmp_loss.item(), batch_size)

                        kd_loss += (tmp_loss * weight)
                total_loss += kd_loss

            # Unified loss computation
            if unified_loss is not None:
                student_module_path_list = config['model']['student']['student_module_paths']
                student_module_io_list = config['model']['student']['student_module_ios']
                teacher_module_path_list = config['model']['teacher']['teacher_module_paths']
                teacher_module_io_list = config['model']['teacher']['teacher_module_ios']
                
                # Get unified loss results
                unified_loss_enabled = config.get('unified_loss', {}).get('loss', {}).get('enabled', [])
                loss_results = unified_loss.forward(
                    student_io_dict, teacher_io_dict, student_module_path_list, 
                    student_module_io_list, teacher_module_path_list, teacher_module_io_list, 
                    size=batch_size)
                
                # Parse results dynamically
                tmp_loss = loss_results[0]  # Total loss is always first
                loss_dict['unified_loss'].update(tmp_loss.item(), batch_size)
                total_loss += tmp_loss
                
                # Parse individual losses based on enabled losses
                loss_idx = 1
                for loss_type in ['mse', 'l1', 'cosine', 'recon']:
                    if loss_type in unified_loss_enabled:
                        if loss_idx < len(loss_results):
                            loss_value = loss_results[loss_idx]
                            loss_dict[f'unified_loss_{loss_type}'].update(loss_value.item(), batch_size)
                            loss_idx += 1

            # Current loss function
            # Loss = alpha * CE + beta * KL + gamma * KDs
            # Default: alpha = 1, beta = 1, gamma = 1
            ce_loss_tmp = criterion(batch_out, batch_y)  # CE loss
            ce_loss += alpha * ce_loss_tmp  # CE loss * alpha
            loss_dict['ce_loss'].update(ce_loss_tmp.item(), batch_size)
            total_loss += ce_loss

            if is_recon_loss:
                z, decoded, mu, logvar = student_io_dict['VIB']['output']
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss_dict['KLD'].update(KLD.item(), batch_size)

                recon_loss = 0.000001*KLD
                loss_dict['recon_loss'].update(recon_loss.item(), batch_size)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] in ['ReduceLROnPlateau', 'MultiStepLR', 'StepLR', 'CyclicLR']:
                pass
            else:
                exp_lr_scheduler.step()
        
        running_loss += (total_loss.item() * batch_size)

        # Calculate accuracy
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

    running_loss /= num_total
    train_acc = (num_correct / num_total) * 100
    logger.info("Accuracy: {}".format(train_acc))
    return running_loss, train_acc, loss_dict


def kd_val_epoch_unified(dev_loader, student, teacher, device, config, 
                        student_forward_hook_manager, teacher_forward_hook_manager, 
                        unified_loss=None):
    """
    Validation epoch with unified loss support
    
    Args:
        dev_loader: Development data loader
        student: Student model
        teacher: Teacher model
        device: Device
        config: Configuration dictionary
        student_forward_hook_manager: Student forward hook manager
        teacher_forward_hook_manager: Teacher forward hook manager
        unified_loss: UnifiedMidLoss instance
    """
    logger.info('Validation with Unified Loss')
    val_loss = 0

    student.eval()
    teacher.eval()

    alpha = float(config['train'].get('alpha', 1))
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    num_correct = 0.0
    running_loss = 0
    loss_dict = dict()
    loss_dict['ce_loss'] = AverageMeter()

    criterions = config.get('criterions', [])
    criterion_key_list = []

    for loss in criterions:
        student_module_path = loss.get('kwargs', {}).get(
            'student_module_path', 'default_student_module_path')
        teacher_module_path = loss.get('kwargs', {}).get(
            'teacher_module_path', 'default_teacher_module_path')
        key = loss.get('key', 'default_key')
        criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
        criterion_key_list.append(criterion_key)
        loss_dict[criterion_key] = AverageMeter()

    # Unified loss tracking for validation
    if unified_loss is not None:
        loss_dict['unified_loss_dev'] = AverageMeter()
        loss_dict['unified_loss_mse_dev'] = AverageMeter()
        loss_dict['unified_loss_cosine_dev'] = AverageMeter()
        unified_loss_enabled = config.get('unified_loss', {}).get('loss', {}).get('enabled', [])
        if 'recon' in unified_loss_enabled:
            loss_dict['unified_loss_recon_dev'] = AverageMeter()
        if 'l1' in unified_loss_enabled:
            loss_dict['unified_loss_l1_dev'] = AverageMeter()

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(dev_loader):
            total_loss = torch.tensor(0.).to(device)
            kd_loss = torch.tensor(0.).to(device)
            ce_loss = torch.tensor(0.).to(device)
            batch_size = batch_x.size(0)
            num_total += batch_size
            if len(batch_x.shape) == 3:
                batch_x = batch_x.squeeze(0).transpose(0, 1)
            batch_x = batch_x.to(device)

            if config["model"]["student"]["name"].startswith("Self"):
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(
                    batch_x)
            else:
                batch_out = student(
                    batch_x)
            student_io_dict = student_forward_hook_manager.pop_io_dict()
            # teacher
            _ = teacher(batch_x)
            teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

            # Check if key exists
            if 'criterions' in config and 'criterion_weights' in config:

                if len(config['criterions']) != len(config['criterion_weights']):
                    raise ValueError(
                        'Number of criterions and criterion_weights must be the same')

                for loss, weight, criterion_key in zip(config['criterions'], config['criterion_weights'], criterion_key_list):
                    weight = float(weight)

                    loss_i = get_mid_level_loss(
                        mid_level_criterion_config=loss)

                    if config['train']['teacher']:
                        tmp_loss = loss_i.forward(student_io_dict,
                                                  teacher_io_dict, batch_y)
                        loss_dict[criterion_key
                                  ].update(tmp_loss.item(), batch_size)

                        kd_loss += (tmp_loss * weight)
                total_loss += kd_loss
                
            # Unified loss computation for validation
            if unified_loss is not None:
                student_module_path_list = config['model']['student']['student_module_paths']
                student_module_io_list = config['model']['student']['student_module_ios']
                teacher_module_path_list = config['model']['teacher']['teacher_module_paths']
                teacher_module_io_list = config['model']['teacher']['teacher_module_ios']
                
                # Get unified loss results
                unified_loss_enabled = config.get('unified_loss', {}).get('loss', {}).get('enabled', [])
                loss_results = unified_loss.forward(
                    student_io_dict, teacher_io_dict, student_module_path_list, 
                    student_module_io_list, teacher_module_path_list, teacher_module_io_list, 
                    size=batch_size)
                
                # Parse results dynamically
                tmp_loss = loss_results[0]  # Total loss is always first
                loss_dict['unified_loss_dev'].update(tmp_loss.item(), batch_size)
                total_loss += tmp_loss
                
                # Parse individual losses based on enabled losses
                loss_idx = 1
                for loss_type in ['mse', 'l1', 'cosine', 'recon']:
                    if loss_type in unified_loss_enabled:
                        if loss_idx < len(loss_results):
                            loss_value = loss_results[loss_idx]
                            loss_dict[f'unified_loss_{loss_type}_dev'].update(loss_value.item(), batch_size)
                            loss_idx += 1

            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)

            ce_loss_tmp = batch_loss  # CE loss
            ce_loss += alpha * ce_loss_tmp  # CE loss * alpha
            loss_dict['ce_loss'].update(ce_loss_tmp.item(), batch_size)
            total_loss += ce_loss

            running_loss += (ce_loss.item() + kd_loss.item()) * batch_size

            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

    accuracy = (num_correct / num_total) * 100
    print("accuracy", accuracy)
    running_loss /= num_total
    print('[VALIDATION] eval_accuracy: ', accuracy)
    return running_loss, accuracy, loss_dict


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))

    return res
