import torch
import torch.nn as nn

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
            # print("teacher_hidden_representation",teacher_hidden_representation.shape)

        # Forward pass with the student model
        # Forward pass with the student model
        student_logits, student_hidden_representation = student(batch_x)
        # print("student_hidden_representation",student_hidden_representation.shape)

        #Soften the student logits by applying softmax first and log() second
        hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_x.size(0)).to(device))
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