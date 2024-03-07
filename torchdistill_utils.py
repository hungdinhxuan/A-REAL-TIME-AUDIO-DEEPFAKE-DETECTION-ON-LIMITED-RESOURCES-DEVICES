
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from kdtoolkit import kd_loss_function, feature_loss_function
from torchdistill.losses.registry import get_mid_level_loss
from contrast.supcontrastloss import SupConLoss, supcon_loss
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('pydub.converter').setLevel(logging.CRITICAL)
logging.getLogger('hydra.core.utils').setLevel(logging.CRITICAL)


def mixup_data(x, y, alpha=0.2):
    batch_size = x.size(0)
    weight = torch.rand(batch_size)
    # Ensure lambda is always between 0.5 and 1
    weight = torch.max(weight, 1 - weight)

    x_mix = x * \
        weight.view(-1, 1) + x[torch.arange(batch_size -
                                            1, -1, -1)] * (1 - weight).view(-1, 1)
    y_mix = y * \
        weight.view(-1, 1) + y[torch.arange(batch_size -
                                            1, -1, -1)] * (1 - weight).view(-1, 1)

    return x_mix, y_mix


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


def self_KD_teacher_val_epoch(dev_loader, model, device, config):
    logger.info('Validation Teacher self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    num_correct = 0.0

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)

            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(
                batch_x)

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
            total_label_loss = batch_loss + spectral_loss + temporal_loss + \
                graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T

            total_loss = total_label_loss

            val_loss += (total_loss.item() * batch_size)

            probabilities = F.softmax(batch_out, dim=1)
            predicted_labels = (probabilities[:, 0] >= 0.5).int()

            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            num_correct += (predicted_labels == batch_y).sum().item()

        accuracy = (num_correct / num_total) * 100
        print("accuracy", accuracy)
        val_loss /= num_total
        # eval_accuracy = (num_correct / num_total) * 100
        print('[VALIDATION] eval_accuracy: ', accuracy)
        return val_loss, accuracy


def self_KD_teacher_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager,   exp_lr_scheduler=None, temperature: float = 3, alpha: float = 0.1, beta: float = 1e-6,  use_amp: bool = True):
    logger.info('Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {}'.format(temperature, alpha, beta))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    running_total_hidden_rep_loss = 0
    running_sup_contrastive_loss = 0

    student.train()
    teacher.eval()
    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    num_total = 0.0

    if not config['train']['teacher']:
        logger.info('No teacher')
        del teacher

    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        logger.info("Current learning rate: {}".format(
            exp_lr_scheduler.get_last_lr()[0]))
    else:
        logger.info("Current learning rate: {}".format(
            optimizer.param_groups[0]['lr']))

    iters = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (batch_x, batch_y) in pbar:

        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(
                batch_x)

            student_io_dict = student_forward_hook_manager.pop_io_dict()

            # Get teacher output
            if config['train']['teacher']:
                with torch.no_grad():
                    pr
                    logits = teacher(batch_x)
                    teacher_io_dict = teacher_forward_hook_manager.pop_io_dict()

            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Multiple loss
            total_mid_level_loss = 0

            # Check if key exists

            if 'criterions' in config and 'criterion_weights' in config:

                if len(config['criterions']) != len(config['criterion_weights']):
                    raise ValueError(
                        'Number of criterions and criterion_weights must be the same')

                # Hard code
                for loss, weight in zip(config['criterions'], config['criterion_weights']):
                    weight = float(weight)
                    # print(loss)
                    # if loss == 'OCKDLoss':
                    #     # Disable dropout

                    loss_i = get_mid_level_loss(
                        mid_level_criterion_config=loss)
                    total_mid_level_loss += (loss_i.forward(
                        student_io_dict, teacher_io_dict) * weight)

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
            kd_spectral_loss = kd_loss_function(
                spectral_output, temp_detach, temperature) * (temperature**2)
            kd_temporal_loss = kd_loss_function(
                temporal_output, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_S = kd_loss_function(
                graph_output_S, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_T = kd_loss_function(
                graph_output_T, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_S = kd_loss_function(
                hs_gal_output_S, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_T = kd_loss_function(
                hs_gal_output_T, temp_detach, temperature) * (temperature**2)

            # Calculate loss (feature loss)
            # We didn't apply backward for final feature
            feature_loss_1 = feature_loss_function(
                middle_feature1, final_feature1.detach())
            feature_loss_2 = feature_loss_function(
                middle_feature2, final_feature2.detach())

            # Calculate total loss
            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss + \
                graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T

            # Total KD loss
            total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_graph_loss_S + \
                kd_graph_loss_T + kd_hs_gal_loss_S + kd_hs_gal_loss_T

            # Total feature loss
            total_feature_loss = feature_loss_1 + feature_loss_2

            # Total loss
            if 'criterions' in config and 'criterion_weights' in config:
                total_loss = (1 - alpha) * total_label_loss + (alpha * total_kd_loss) + \
                    (beta * total_feature_loss) + total_mid_level_loss
            else:
                total_loss = (1 - alpha) * total_label_loss + \
                    (alpha * total_kd_loss) + (beta * total_feature_loss)

            if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
                sup_weights = config["train"].get(
                    "sup_contrastive_loss_weight", [0.07, 0.07])
                sup_loss = SupConLoss(
                    temperature=sup_weights[0], base_temperature=sup_weights[1])
                sup_mode = config["train"].get(
                    "sup_contrastive_mode", "supcon")

                if sup_mode == "supcon":
                    sup_loss = sup_loss.forward(
                        student_hidden_representation, batch_y)
                else:
                    sup_loss = sup_loss.forward(student_hidden_representation)

                total_loss += sup_loss

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] in ['ReduceLROnPlateau', 'MultiStepLR', 'StepLR']:
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()

        running_loss += (total_loss.item() * batch_size)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)

        if 'criterions' in config and 'criterion_weights' in config:
            running_total_hidden_rep_loss += (
                total_mid_level_loss.item() * batch_size)
        if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
            running_sup_contrastive_loss += (sup_loss.item() * batch_size)
    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss, running_total_hidden_rep_loss, running_sup_contrastive_loss


def kd_train_epoch(train_loader, student, teacher, optimizer, device, scaler, config, student_forward_hook_manager, teacher_forward_hook_manager,  exp_lr_scheduler=None,  use_amp: bool = True):
    logger.info('Training KD')
    running_loss = 0

    student.train()
    teacher.eval()

    num_correct = 0.0

    mixup = config["train"].get("mixup", False)

    if "alpha" not in config['train']:
        forward_target = True
    else:
        forward_target = False
        alpha = float(config['train']['alpha'])

        if 'beta' in config['train']:
            beta = float(config['train']['beta'])
        else:
            beta = 0.5

    weight = torch.FloatTensor(config['train'].get(
        'cross_entropy_loss_weight', [0.1, 0.9])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    dot = config['train'].get('dot', False)

    num_total = 0.0

    if not config['train']['teacher']:
        logger.info('No teacher')
        del teacher

    if exp_lr_scheduler is not None and config['learning_rate_scheduler']['name'] != 'ReduceLROnPlateau':
        logger.info("Current learning rate: {}".format(
            exp_lr_scheduler.get_last_lr()[0]))
    else:
        logger.info("Current learning rate: {}".format(
            optimizer.param_groups[0]['lr']))

    iters = len(train_loader)
    # Create a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (batch_x, batch_y) in pbar:

        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # Multiple loss
            total_loss = torch.tensor(0.).to(device)
            kd_loss = torch.tensor(0.).to(device)
            ce_loss = torch.tensor(0.).to(device)
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            if len(batch_x.shape) == 3:
                batch_x = batch_x.squeeze(0).transpose(0, 1)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Mixup data
            if mixup:
                alpha = 0.1
                lam = np.random.beta(alpha, alpha)
                lam = torch.tensor(lam, requires_grad=False)
                index = torch.randperm(len(batch_y))
                batch_x = lam*batch_x + (1-lam)*batch_x[index, :]
                batch_y_b = batch_y[index]

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

                for loss, weight in zip(config['criterions'], config['criterion_weights']):
                    weight = float(weight)
                    # print("Current loss: ", loss)
                    # if loss['key'] == 'OCKDLoss':
                    #     student.module.ssl_model.model.eval()
                    # else:
                    #     student.module.ssl_model.model.train()

                    loss_i = get_mid_level_loss(
                        mid_level_criterion_config=loss)

                    if forward_target:
                        total_loss += (loss_i.forward(student_io_dict,
                                       teacher_io_dict, batch_y) * weight)
                        kd_loss += (loss_i.forward(student_io_dict,
                                    teacher_io_dict, batch_y) * weight)
                    else:
                        total_loss += (loss_i.forward(student_io_dict,
                                       teacher_io_dict) * weight)
                        kd_loss += (loss_i.forward(student_io_dict,
                                    teacher_io_dict) * weight)

            if not forward_target or dot:
                alpha = 1
                # CE loss
                batch_loss = criterion(batch_out, batch_y)
                total_loss += (alpha * batch_loss)
                ce_loss += (alpha * batch_loss)

                # ## Total loss + KL divergence loss
                # kl_loss = DistillKL(T=config["train"].get("T", 2))
                # total_loss += (beta * kl_loss(batch_out, t_logits))
                if mixup:
                    batch_loss_b = criterion(batch_out, batch_y_b)
                    total_loss += (lam * total_loss + (1-lam) * batch_loss_b)
                    ce_loss += (lam * ce_loss + (1-lam) * batch_loss_b)

            if "sup_contrastive" in config["train"] and config["train"]["sup_contrastive"]:
                sup_weights = config["train"].get(
                    "sup_contrastive_loss_weight", [0.1, 0.07])
                sup_loss = SupConLoss(
                    temperature=sup_weights[0], base_temperature=sup_weights[1])
                sup_mode = config["train"].get(
                    "sup_contrastive_mode", "supcon")

                if sup_mode == "supcon":
                    def sim_metric_seq(mat1, mat2): return torch.bmm(
                        mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)

                    # Get embedding student (hard-code here)
                    emb = student_io_dict['LL']['output']
                    # reshape the emb to match the supcon loss format
                    # emb = emb.unsqueeze(1)
                    # emb = emb.unsqueeze(-1)

                    # feats = feats.unsqueeze(1)

                    # sup_loss = supcon_loss(emb,
                    #                        batch_y)

                    sup_loss = sup_loss.forward(
                        emb, batch_y)
                else:
                    sup_loss = sup_loss.forward(student_hidden_representation)
                total_loss += sup_loss

        # Scaler
        if not dot:
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)

            kd_loss.backward(retain_graph=True)
            optimizer.step_kd()
            optimizer.zero_grad(set_to_none=True)
            ce_loss.backward()
            optimizer.step()
        # Update LR
        if exp_lr_scheduler is not None:
            if config['learning_rate_scheduler']['name'] == 'CosineAnnealingWarmRestarts':
                exp_lr_scheduler.step(epoch + i / iters)
            elif config['learning_rate_scheduler']['name'] == 'ReduceLROnPlateau' or config['learning_rate_scheduler']['name'] == 'MultiStepLR' or config['learning_rate_scheduler']['name'] == 'StepLR':
                # Update learning rate scheduler in validation so do nothing here
                pass
            else:
                exp_lr_scheduler.step()
        if not dot:
            running_loss += (total_loss.item() * batch_size)
        else:
            # logger.info("KD loss: {} - CE loss: {}".format(kd_loss.item(), ce_loss.item()))
            running_loss += (ce_loss.item() + kd_loss.item()) * batch_size
        _, batch_pred = batch_out.max(dim=1)
        # batch_y = batch_y.view(-1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

    running_loss /= num_total
    logger.info("Accuracy: {}".format((num_correct / num_total) * 100))
    return running_loss


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

            # Calculate EER
            # for x, y in zip(batch_x, batch_y):
            #     if y == 1:
            #         bona_scores.append(x)
            #     else:
            #         spoof_scores.append(x)

        # eer_cm, th = em.compute_eer(bona_scores, spoof_scores) * 100
        # logger.info("EER: {}% - Threshold: {}".format(eer_cm , th))
        accuracy = (num_correct / num_total) * 100
        print("accuracy", accuracy)
        val_loss /= num_total
        print('[VALIDATION] eval_accuracy: ', accuracy)
        return val_loss, accuracy
