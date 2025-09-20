
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from kdtoolkit import kd_loss_function, feature_loss_function
from torchdistill.losses.registry import get_mid_level_loss
from contrast.supcontrastloss import SupConLoss, supcon_loss
import numpy as np
from utils import AverageMeter
#from losses import MSELoss, CosineLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('pydub.converter').setLevel(logging.CRITICAL)
logging.getLogger('hydra.core.utils').setLevel(logging.CRITICAL)


def kd_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager,  epoch, exp_lr_scheduler=None,  use_amp: bool = True, standard_mid_loss=None):
    logger.info('Training KD')
    running_loss = 0
    running_standard_mid_loss = 0

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
        # loss_dict[f"{loss['key']}_{loss['kwargs']['student_module_path']}_{loss['kwargs']['teacher_module_path']}"] = 0

        student_module_path = loss.get('kwargs', {}).get(
            'student_module_path', 'default_student_module_path')
        teacher_module_path = loss.get('kwargs', {}).get(
            'teacher_module_path', 'default_teacher_module_path')
        key = loss.get('key', 'default_key')
        criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
        criterion_key_list.append(criterion_key)
        loss_dict[criterion_key] = AverageMeter()
    
    loss_dict['standard_mid_loss'] = AverageMeter()
    loss_dict['standard_mid_loss_mse'] = AverageMeter()
    loss_dict['standard_mid_loss_cosine'] = AverageMeter()
    loss_dict['standard_mid_loss_recon'] = AverageMeter()

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
                batch_out, s_layer_results = student(
                    batch_x, layerwise=True)
                    

            student_io_dict = student_forward_hook_manager.pop_io_dict()

            # Get teacher output
            if config['train']['teacher']:
                with torch.no_grad():
                    t_logits, t_layer_results = teacher(batch_x, layerwise=True)
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

            if standard_mid_loss is not None:
                tmp_loss, tmp_mse_loss, tmp_cosine_loss, tmp_recon_loss = standard_mid_loss.forward(s_layer_results, t_layer_results, size=batch_size)
                loss_dict['standard_mid_loss'].update(tmp_loss.item(), batch_size)
                loss_dict['standard_mid_loss_mse'].update(tmp_mse_loss.item(), batch_size)
                loss_dict['standard_mid_loss_cosine'].update(tmp_cosine_loss.item(), batch_size)
                loss_dict['standard_mid_loss_recon'].update(tmp_recon_loss.item(), batch_size)
                total_loss += tmp_loss
            #     student_module_path_list = config['model']['student']['student_module_paths']
            #     student_module_io_list = config['model']['student']['student_module_ios']
            #     teacher_module_path_list = config['model']['teacher']['teacher_module_paths']
            #     teacher_module_io_list = config['model']['teacher']['teacher_module_ios']
            #     #print(f"Length student feat extracted: {len(student.front_end.extract_layer_results(batch_x))}")
            #     tmp_loss, tmp_mse_loss, tmp_cosine_loss, tmp_recon_loss = standard_mid_loss.forward(student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, size=batch_size)
            #     loss_dict['standard_mid_loss'].update(tmp_loss.item(), batch_size)
            #     total_loss += tmp_loss

            #     loss_dict['standard_mid_loss_mse'].update(tmp_mse_loss.item(), batch_size)
            #     loss_dict['standard_mid_loss_cosine'].update(tmp_cosine_loss.item(), batch_size)
            #     loss_dict['standard_mid_loss_recon'].update(tmp_recon_loss.item(), batch_size)


            # Current loss function
            # Loss = alpha * CE + beta * KL + gamma * KDs
            # Default: alpha = 1, beta = 1, gamma = 1
            ce_loss_tmp = criterion(batch_out, batch_y)  # CE loss
            ce_loss += alpha * ce_loss_tmp  # CE loss * alpha
            loss_dict['ce_loss'].update(ce_loss_tmp.item(), batch_size)
            total_loss += ce_loss

            if is_recon_loss:
                z, decoded, mu, logvar = student_io_dict['VIB']['output']
                # feats_w2v = student_io_dict['LL']['output']

                # BCE = F.binary_cross_entropy(torch.sigmoid(
                #     decoded), torch.sigmoid(feats_w2v), reduction='sum')
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                # loss_dict['BCE'].update(BCE.item(), batch_size)
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

            if config["model"]["student"]["name"].startswith("Self"):
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = model(
                    batch_x)
            else:
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


def kd_val_epoch_advanced(dev_loader, student, teacher, device,  config, student_forward_hook_manager, teacher_forward_hook_manager, standard_mid_loss=None):
    logger.info('Validation ----')
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
        # loss_dict[f"{loss['key']}_{loss['kwargs']['student_module_path']}_{loss['kwargs']['teacher_module_path']}"] = 0

        student_module_path = loss.get('kwargs', {}).get(
            'student_module_path', 'default_student_module_path')
        teacher_module_path = loss.get('kwargs', {}).get(
            'teacher_module_path', 'default_teacher_module_path')
        key = loss.get('key', 'default_key')
        criterion_key = f"{key}_{student_module_path}_{teacher_module_path}"
        criterion_key_list.append(criterion_key)
        loss_dict[criterion_key] = AverageMeter()

    loss_dict['standard_mid_loss_dev'] = AverageMeter()
    loss_dict['standard_mid_loss_mse_dev'] = AverageMeter()
    loss_dict['standard_mid_loss_cosine_dev'] = AverageMeter()
    loss_dict['standard_mid_loss_recon_dev'] = AverageMeter()

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
                batch_out, s_layer_results = student(
                    batch_x, layerwise=True)
            student_io_dict = student_forward_hook_manager.pop_io_dict()
            # teacher
            _, t_layer_results = teacher(batch_x, layerwise=True)
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
            if standard_mid_loss is not None:
                tmp_loss, tmp_mse_loss, tmp_cosine_loss, tmp_recon_loss = standard_mid_loss.forward(s_layer_results, t_layer_results, size=batch_size)
                loss_dict['standard_mid_loss_dev'].update(tmp_loss.item(), batch_size)
                loss_dict['standard_mid_loss_mse_dev'].update(tmp_mse_loss.item(), batch_size)
                loss_dict['standard_mid_loss_cosine_dev'].update(tmp_cosine_loss.item(), batch_size)
                loss_dict['standard_mid_loss_recon_dev'].update(tmp_recon_loss.item(), batch_size)
                total_loss += tmp_loss
            # if standard_mid_loss is not None:
            #     student_module_path_list = config['model']['student']['student_module_paths']
            #     student_module_io_list = config['model']['student']['student_module_ios']
            #     teacher_module_path_list = config['model']['teacher']['teacher_module_paths']
            #     teacher_module_io_list = config['model']['teacher']['teacher_module_ios']
            #     #print(f"Length student feat extracted: {len(student.front_end.extract_layer_results(batch_x))}")
            #     tmp_loss, tmp_mse_loss, tmp_cosine_loss, tmp_recon_loss = standard_mid_loss.forward(student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, size=batch_size)
            #     loss_dict['standard_mid_loss_dev'].update(tmp_loss.item(), batch_size)
            #     loss_dict['standard_mid_loss_mse_dev'].update(tmp_mse_loss.item(), batch_size)
            #     loss_dict['standard_mid_loss_cosine_dev'].update(tmp_cosine_loss.item(), batch_size)
            #     loss_dict['standard_mid_loss_recon_dev'].update(tmp_recon_loss.item(), batch_size)
            #     total_loss += tmp_loss

            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)

            # val_loss += (batch_loss.item() * batch_size)

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