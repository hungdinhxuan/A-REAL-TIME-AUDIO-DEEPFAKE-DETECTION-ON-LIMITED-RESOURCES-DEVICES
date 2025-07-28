
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torchdistill.losses.registry import get_mid_level_loss
import numpy as np
from utils import AverageMeter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('pydub.converter').setLevel(logging.CRITICAL)
logging.getLogger('hydra.core.utils').setLevel(logging.CRITICAL)

def kd_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager,  epoch, exp_lr_scheduler=None,  use_amp: bool = True, weighted_views: Dict[str, float] = {
        '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0}):
    logger.info('Training KD')
    running_loss = 0

    student.train()
    teacher.eval()

    num_correct = 0.0

    forward_target = "alpha" not in config['train']
    alpha = float(config['train'].get('alpha', 1))
    beta = float(config['train'].get('beta', 0.5))
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
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
    pbar = tqdm(train_loader)
    # loss list for monitoring
    loss_dict = dict()
    loss_dict['ce_loss'] = AverageMeter()

    criterions = config.get('criterions', [])
    criterion_key_list = []


    for loss in criterions:
        # loss_dict[f"{loss['key']}_{loss['kwargs']['student_module_path']}_{loss['kwargs']['teacher_module_path']}"] = 0

        student_module_path = loss.get('kwargs', {}).get(
            'student_module_path', 'default_student_module_path')
        teacher_module_path = loss.get('kwargs', {}).get(
            'teacher_module_path', 'default_teacher_module_path')
        key = loss.get('key', 'default_key')
        criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
        criterion_key_list.append(criterion_key)
        loss_dict[criterion_key] = AverageMeter()

    # Initialize metrics for each view
    running_losses = {f'view_{v}': 0.0 for v in weighted_views.keys()}
    running_corrects = {f'view_{v}': 0 for v in weighted_views.keys()}
    running_samples = {f'view_{v}': 0 for v in weighted_views.keys()}
    
    for batch in pbar:
        total_loss = 0.0
        batch_metrics = {}
        for view_idx, (inputs, labels) in batch.items():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                # Multiple loss
                view = str(view_idx)  # Convert view index to string
                
                total_loss = torch.tensor(0.).to(device)
                kd_loss = torch.tensor(0.).to(device)
                ce_loss = torch.tensor(0.).to(device)
            
                batch_size = batch_x.size(0)
                
                batch_x = batch_x.to(device)
                if len(batch_x.shape) == 3:
                    batch_x = batch_x.squeeze(0).transpose(0, 1)

                batch_y = batch_y.view(-1).type(torch.int64).to(device)

                num_total += batch_size
                batch_x = batch_x.to(device)

            
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
                            mid_level_criterion_config=loss) # try to find the class match with name defined in config

                        if config['train']['teacher']:
                            tmp_loss = loss_i.forward(student_io_dict,
                                                    teacher_io_dict, size=batch_size)
                            loss_dict[criterion_key
                                    ].update(tmp_loss.item(), batch_size)

                            kd_loss += (tmp_loss * weight)
                    total_loss += kd_loss

                # Current loss function
                # Loss = alpha * CE + beta * KL + gamma * KDs
                # Default: alpha = 1, beta = 1, gamma = 1
                ce_loss_tmp = criterion(batch_out, batch_y)  # CE loss
                ce_loss += alpha * ce_loss_tmp  # CE loss * alpha
                loss_dict['ce_loss'].update(ce_loss_tmp.item(), batch_size)
                total_loss += ce_loss
                
            ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
            view = str(view_idx)  # Convert view index to string

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.view(-1).type(torch.int64).to(device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels) * weighted_views[view]

            # Calculate metrics
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()

            # Update running metrics
            running_losses[f'view_{view}'] += loss.item() * batch_size
            running_corrects[f'view_{view}'] += correct
            running_samples[f'view_{view}'] += batch_size

            # Accumulate total loss
            total_loss += loss

            # Calculate batch metrics for progress bar
            batch_metrics[f'loss_view_{view}'] = loss.item()
            batch_metrics[f'acc_view_{view}'] = correct / batch_size


        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                # logger.info("Updating learning rate in training")
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] in ['ReduceLROnPlateau', 'MultiStepLR', 'StepLR', 'CyclicLR']:
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()
        
        running_loss += (total_loss.item() * batch_size)

        # Calculate accuracy
        _, batch_pred = batch_out.max(dim=1)
        # batch_y = batch_y.view(-1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

    running_loss /= num_total
    train_acc = (num_correct / num_total) * 100
    logger.info("Accuracy: {}".format(train_acc))
    return running_loss, train_acc, loss_dict


def kd_val_epoch(dev_loader, model, device, config):
    logger.info('Validation ----')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    num_correct = 0.0
    bona_scores = []
    spoof_scores = []

    loss_dict = dict()
    loss_dict['ce_loss'] = AverageMeter()

    criterions = config.get('criterions', [])
    criterion_key_list = []

    for loss in criterions:
        # loss_dict[f"{loss['key']}_{loss['kwargs']['student_module_path']}_{loss['kwargs']['teacher_module_path']}"] = 0

        student_module_path = loss.get('kwargs', {}).get(
            'student_module_path', 'default_student_module_path')
        teacher_module_path = loss.get('kwargs', {}).get(
            'teacher_module_path', 'default_teacher_module_path')
        key = loss.get('key', 'default_key')
        criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
        criterion_key_list.append(criterion_key)
        loss_dict[criterion_key] = AverageMeter()

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            if len(batch_x.shape) == 3:
                batch_x = batch_x.squeeze(0).transpose(0, 1)
            batch_x = batch_x.to(device)

       
            batch_out = model(
                batch_x)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)

            val_loss += (batch_loss.item() * batch_size)

            # probabilities = F.softmax(batch_out, dim=1)
            # predicted_labels = (probabilities[:, 0] >= 0.5).int()

            # num_correct += (predicted_labels == batch_y).sum().item()
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()

        # eer_cm, th = em.compute_eer(bona_scores, spoof_scores) * 100
        # logger.info("EER: {}% - Threshold: {}".format(eer_cm , th))
        accuracy = (num_correct / num_total) * 100
        print("accuracy", accuracy)
        val_loss /= num_total
        print('[VALIDATION] eval_accuracy: ', accuracy)
        return val_loss, accuracy


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

