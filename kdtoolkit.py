import torch
import torch.nn as nn
import logging
import os
from concurrent.futures import ThreadPoolExecutor

def load_tensor(utt):
    return torch.load(os.path.join('teacher_cosine_emb', f'{utt}.pt'))

def kd_loss_function(output, target_output, temperature):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

def train_knowledge_distillation(teacher, student, train_loader, optimizer, T, soft_target_loss_weight, ce_loss_weight, device):
    # ce_loss = nn.CrossEntropyLoss()
    print('Training student with knowledge distillation. T: {}, soft_target_loss_weight: {}, ce_loss_weight: {}'.format(T, soft_target_loss_weight, ce_loss_weight))
    #set objective (Loss) functions
    running_loss = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    num_total = 0.0

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
    return running_loss


def train_kd_cosine_loss(teacher, student, train_loader, optimizer,  hidden_rep_loss_weight, ce_loss_weight, device):
    
    print('Training student with KD cosine loss. hidden_rep_loss_weight: {}, ce_loss_weight: {}'.format(hidden_rep_loss_weight, ce_loss_weight))
    #set objective (Loss) functions
    cosine_loss = nn.CosineEmbeddingLoss()
    running_loss = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    num_total = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size
        optimizer.zero_grad()

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        # Forward pass with the teacher model and keep only the hidden representation
        with torch.no_grad():
            _, teacher_hidden_representation = teacher(batch_x)
            

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
    return running_loss


def self_KD_Dropout_train_epoch(train_loader, teacher, student, optimizer, device, scaler, lr_scheduler, temperature=3, alpha=0.01, beta=1e-6, hidden_rep_loss_weight=0.15, hlambda=0.1, use_amp = True):
    logging.log(logging.INFO, 'Training self KD Dropout with temperature = {} and alpha = {} and beta = {} and hidden_rep_loss_weight = {} and hlambda = {}'.format(temperature, alpha, beta , hidden_rep_loss_weight, hlambda))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    running_total_kl_loss = 0
    # kl_div_loss = nn.KLDivLoss(reduction='none')
    
    cosine_loss = nn.CosineEmbeddingLoss()
    student.train()
    teacher.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0


    for batch_x, batch_y in train_loader:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, batch_out2, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Get teacher output
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(batch_x)
            
            student_hidden_representation = student_hidden_representation.view(batch_size, -1)
            teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)

            # Compute the KL divergence loss
            loss_kl_1 = nn.functional.kl_div(torch.nn.functional.log_softmax(batch_out, dim=1), torch.nn.functional.softmax(batch_out2, dim=1), reduction="batchmean")
            
            loss_kl_2 = nn.functional.kl_div(torch.nn.functional.log_softmax(batch_out2, dim=1), torch.nn.functional.softmax(batch_out, dim=1), reduction="batchmean")
            
            total_kl_loss = loss_kl_1 + loss_kl_2

            # Now pass the reshaped tensors to cosine_loss
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))

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
            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss + hidden_rep_loss_weight * hidden_rep_loss + hlambda * total_kl_loss

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        
        scaler.step(optimizer)
        scaler.update()

        
        running_loss += (total_loss.item() * batch_size)
        # print('[TRAINING] running_loss: ', running_loss)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        # print('[TRAINING] running_total_label_loss: ', running_total_label_loss)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        # print('[TRAINING] running_total_kd_loss: ', running_total_kd_loss)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)
        # print('[TRAINING] running_total_feature_loss: ', running_total_feature_loss)
        running_total_kl_loss += (total_kl_loss.item() * batch_size)
        # print('[TRAINING] running_total_kl_loss: ', running_total_kl_loss)
    
    lr_scheduler.step()
    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    running_total_kl_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss, running_total_kl_loss

def self_KD_train_epoch(train_loader, model, optimizer, device, scaler, temperature=3, alpha=0.1, beta=1e-6, use_amp = True):
    logging.log(logging.INFO, 'Training self KD with temperature = {} and alpha = {} and beta = {}'.format(temperature, alpha, beta))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0

    

    for batch_x, batch_y in train_loader:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = model(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

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
            # print('[TRAINING] total_label_loss: ', total_label_loss)

            # Total KD loss
            total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_graph_loss_S + kd_graph_loss_T + kd_hs_gal_loss_S + kd_hs_gal_loss_T
            # print('[TRAINING] total_kd_loss: ', total_kd_loss)

            # Total feature loss
            total_feature_loss = feature_loss_1 + feature_loss_2
            # print('[TRAINING] total_feature_loss: ', total_feature_loss)

            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss

            # print('[TRAINING] total_loss: ', total_loss)

        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
            
        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        running_loss += (total_loss.item() * batch_size)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)
    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss

def self_KD_teacher_train_epoch(train_loader, student, teacher, optimizer, device, scaler, temperature=3, alpha=0.1, beta=1e-6, hidden_rep_loss_weight=0.5, use_amp = True):
    logging.log(logging.INFO, 'Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {} and hidden_rep_loss_weight = {}'.format(temperature, alpha, beta, hidden_rep_loss_weight))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    cosine_loss = nn.CosineEmbeddingLoss()
    student.train()
    teacher.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    

    for batch_x, batch_y in train_loader:
        # Get teacher output
        with torch.no_grad():
            _, teacher_hidden_representation = teacher(batch_x)
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            
            student_hidden_representation = student_hidden_representation.view(batch_size, -1)
            teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)

            # Now pass the reshaped tensors to cosine_loss
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))

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
            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss + hidden_rep_loss_weight * hidden_rep_loss

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        
        running_loss += (total_loss.item() * batch_size)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)

    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss

def self_KD_teacher2_train_epoch(train_loader, student, teacher, optimizer, device, scaler, temperature=3, alpha=0.1, beta=1e-6, hidden_rep_loss_weight=0.5, use_amp = True):
    logging.log(logging.INFO, 'Training self KD + teacher 2 cosine with temperature = {} and alpha = {} and beta = {} and hidden_rep_loss_weight = {}'.format(temperature, alpha, beta, hidden_rep_loss_weight))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    cosine_loss = nn.CosineEmbeddingLoss()
    student.train()
    teacher.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0

    for batch_x, batch_y in train_loader:
        # Get teacher output
        with torch.no_grad():
            _, teacher_hidden_representation = teacher(batch_x)
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            
            
            student_hidden_representation = student_hidden_representation.view(batch_size, -1)
            teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)

            # Now pass the reshaped tensors to cosine_loss
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)

            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Calculate KD loss
            temp = batch_out / temperature
            temp = torch.softmax(temp, dim=1)
            temp_detach = temp.detach()
            kd_spectral_loss = kd_loss_function(spectral_output, temp_detach, temperature) * (temperature**2)
            kd_temporal_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
  
            kd_hs_gal_loss_S = kd_loss_function(hs_gal_output_S, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_T = kd_loss_function(hs_gal_output_T, temp_detach, temperature) * (temperature**2)

            # Calculate loss (feature loss)
            # We didn't apply backward for final feature
            feature_loss_1 = feature_loss_function(middle_feature1, final_feature1.detach())
            feature_loss_2 = feature_loss_function(middle_feature2, final_feature2.detach())

            # Calculate total loss
            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss  + hs_gal_loss_S + hs_gal_loss_T

            # Total KD loss
            total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_hs_gal_loss_S + kd_hs_gal_loss_T

            # Total feature loss
            total_feature_loss = feature_loss_1 + feature_loss_2

            # Total loss
            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss + hidden_rep_loss_weight * hidden_rep_loss

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        
        running_loss += (total_loss.item() * batch_size)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)

    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss

def self_KD_val_epoch(dev_loader, model,  device):
    logging.log(logging.INFO, 'Validation self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_correct = 0.0
    num_total = 0.0


    with torch.inference_mode():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = model(batch_x)
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

            batch_pred = torch.sigmoid(batch_out)
            batch_pred_label = 1 if batch_pred > 0.5 else 0
            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()


            val_loss += (total_loss.item() * batch_size)
        val_loss /= num_total
        eval_accuracy = (num_correct / num_total) * 100
        print('[VALIDATION] eval_accuracy: ', eval_accuracy)
        return val_loss, eval_accuracy

def self_KD_teacher_val_epoch(dev_loader, model, device, kd_method='self_KD_Teacher'):
    logging.log(logging.INFO, 'Validation Teacher self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    num_correct = 0.0

    with torch.inference_mode():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            if kd_method == 'self_KD_Teacher':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            else:
                batch_out, batch_out2, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            #true label
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

            # # Return [batch_size, 2]
            # batch_pred = torch.sigmoid(batch_out)

            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            # # In sigmoid it will return like [0.1, 0.9] where 0.1 is fake and 0.9 is genuine. However, we just care about fake value
            # # So we just take the first value and compare it with 0.5 to get the label
            # # If first value > 0.5 then label = 1 else label = 0
            
            
            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            # batch_pred_label = (batch_pred > 0.5).int()
            
            # # print(batch_pred_label)
            
            # print(batch_pred_label.shape)
            
            # print(batch_y.shape)

            # num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            val_loss += (total_loss.item() * batch_size)
        val_loss /= num_total
        # eval_accuracy = (num_correct / num_total) * 100
        # print('[VALIDATION] eval_accuracy: ', eval_accuracy)
        return val_loss

def self_KD_teacher2_val_epoch(dev_loader, model, device, kd_method='self_KD_Teacher'):
    logging.log(logging.INFO, 'Validation Teacher self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0


    with torch.inference_mode():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            if kd_method == 'self_KD_Teacher':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            else:
                batch_out, batch_out2, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)
            # graph_loss_S = criterion(graph_output_S, batch_y)
            # graph_loss_T = criterion(graph_output_T, batch_y)
            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss  + hs_gal_loss_S + hs_gal_loss_T
            
            total_loss = total_label_loss


            val_loss += (total_loss.item() * batch_size)
        val_loss /= num_total
        return val_loss

# This is version 2 of self_KD_teacher_train_epoch where we will load teacher_hidden_representation from the file
def self_KD_teacher_train_epoch_v2(train_loader, student, optimizer, device, scaler, temperature=3, alpha=0.1, beta=1e-6, hidden_rep_loss_weight=0.5, use_amp = True):
    logging.log(logging.INFO, 'Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {} and hidden_rep_loss_weight = {}'.format(temperature, alpha, beta, hidden_rep_loss_weight))
    running_loss = 0
    running_total_label_loss = 0
    running_total_kd_loss = 0
    running_total_feature_loss = 0
    cosine_loss = nn.CosineEmbeddingLoss()
    student.train()
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0

    for batch_x,batch_y, teacher_hidden_representation in train_loader:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)


            teacher_hidden_representation = teacher_hidden_representation.to(device)
            student_hidden_representation = student_hidden_representation.view(batch_size, -1)
            teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)

            # Now pass the reshaped tensors to cosine_loss
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))

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
            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss + hidden_rep_loss_weight * hidden_rep_loss

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        
        running_loss += (total_loss.item() * batch_size)
        running_total_label_loss += (total_label_loss.item() * batch_size)
        running_total_kd_loss += (total_kd_loss.item() * batch_size)
        running_total_feature_loss += (total_feature_loss.item() * batch_size)

    running_loss /= num_total
    running_total_feature_loss /= num_total
    running_total_label_loss /= num_total
    running_total_kd_loss /= num_total
    return running_loss, running_total_label_loss, running_total_kd_loss, running_total_feature_loss

def train_kd_mse_loss(teacher, student, train_loader, optimizer, feature_map_weight, ce_loss_weight, device):
    
    print('Training student with KD mse loss. feature_map_weight: {}, ce_loss_weight: {}'.format(feature_map_weight, ce_loss_weight))
    #set objective (Loss) functions
    mse_loss = nn.MSELoss()
    running_loss = 0
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode
    num_total = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
        batch_size = batch_x.size(0)
        num_total += batch_size
        optimizer.zero_grad()

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        # Forward pass with the teacher model and keep only the hidden representation
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
        
        # print("loss",loss)

        loss.backward()
        optimizer.step()

        running_loss += (loss.item() * batch_size)
    running_loss /= num_total
    return running_loss