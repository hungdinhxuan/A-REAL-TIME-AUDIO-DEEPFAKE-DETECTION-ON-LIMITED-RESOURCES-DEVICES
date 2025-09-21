import torch.nn.functional as F
from torch import nn
import torch
# Define a linear layer to transform from 256 to 1024 dimensions
from autoencoders import ShallowAutoencoder, DeepAutoencoder

class StandardMidLoss(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.5)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)

    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # import sys
        # sys.exit()
        
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_maps)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_maps.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        
        # hard code for weight
        # mse_loss = mse_loss * 0.5
        # #cosine_loss = cosine_loss * 0.5
        
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss
        
        return loss, mse_loss, cosine_loss

class StandardMidLoss_v2(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    The projection layer is a linear layer with the same dimension as the student and teacher
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = nn.Linear(1024, 768, device=device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
                                                                                                           
    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        
        teacher_feature_proj = self.t_projection(teacher_feature_maps)
        teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss 
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, cosine_loss


class StandardMidLoss_v2_hf(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    The projection layer is a linear layer with the same dimension as the student and teacher
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = nn.Linear(1024, 768, device=device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
                                                                                                           
    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        student_feature_maps = student_feature_maps.permute(0, 2, 1, 3)
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        
        teacher_feature_proj = self.t_projection(teacher_feature_maps)
        teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss 
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, cosine_loss

class StandardMidLoss_v3(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    This version uses the ShallowAutoencoder to project the student and teacher features
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = ShallowAutoencoder(input_dim=1024, latent_dim=768, use_bias=True).to(device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
        self.recon_loss_weight = kwargs.get('recon_loss_weight', 0.0001)

    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
     
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
           
        teacher_feature_proj, teacher_feature_recon = self.t_projection(teacher_feature_maps)
        #teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        recon_loss = self.mse_loss(teacher_feature_maps, teacher_feature_recon)
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        scaled_recon_loss = recon_loss * self.recon_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss + scaled_recon_loss
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, cosine_loss, recon_loss

class StandardMidLoss_v2_deep(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    This version uses the DeepAutoencoder to project the student and teacher features
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = DeepAutoencoder(dim=[1024, 768, 768], use_bias=True).to(device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
        self.recon_loss_weight = kwargs.get('recon_loss_weight', 0.0001)

    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
     
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
           
        teacher_feature_proj, _ = self.t_projection(teacher_feature_maps)
        #teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        #recon_loss = self.mse_loss(teacher_feature_maps, teacher_feature_recon)
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        #scaled_recon_loss = recon_loss * self.recon_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss #+ scaled_recon_loss
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, cosine_loss #, recon_loss

class StandardMidLoss_v4(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    This version uses the ShallowAutoencoder to project the student and teacher features
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = ShallowAutoencoder(input_dim=1024, latent_dim=768, use_bias=True).to(device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
        self.recon_loss_weight = kwargs.get('recon_loss_weight', 0.0001)

    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
     
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
           
        teacher_feature_proj, teacher_feature_recon = self.t_projection(teacher_feature_maps)
        #teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        #cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        recon_loss = self.mse_loss(teacher_feature_maps, teacher_feature_recon)
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        #scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        scaled_recon_loss = recon_loss * self.recon_loss_weight
        
        loss = scaled_mse_loss + scaled_recon_loss
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, recon_loss

class StandardMidLoss_v5(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    This version uses the ShallowAutoencoder to project the student and teacher features
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = ShallowAutoencoder(input_dim=1024, latent_dim=768, use_bias=True).to(device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
        self.recon_loss_weight = kwargs.get('recon_loss_weight', 0.0001)

    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
     
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
           
        teacher_feature_proj, teacher_feature_recon = self.t_projection(teacher_feature_maps)
        #teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        recon_loss = self.mse_loss(teacher_feature_maps, teacher_feature_recon)
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        scaled_recon_loss = recon_loss * self.recon_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss + scaled_recon_loss
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, cosine_loss, recon_loss


class StandardMidLoss_v5_hf(nn.Module):
    """
    A loss module for the Knowledge Distillation from two embedding features
    This is the same as StandardMidLoss, but we have one more projection layer to make the student and teacher have the same dimension
    This version uses the ShallowAutoencoder to project the student and teacher features
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        # Initialize projection for student and teacher
        #self.student_projection = nn.Linear(num_student_layers, 1) # (num_layers, 1)
        #self.teacher_projection = nn.Linear(num_teacher_layers, 1) # (num_layers, 1)
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.device = device
        self.t_projection = ShallowAutoencoder(input_dim=1024, latent_dim=768, use_bias=True).to(device)
        self.s_layers = s_layers
        #self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
        self.recon_loss_weight = kwargs.get('recon_loss_weight', 0.0001)

    def forward(self, student_io_dict, teacher_io_dict, student_module_path_list, student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs):
        # Concatenate student and teacher feature maps
        self.size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path != 'backend':
                # Mapping student_module_io 'False:True' to 'output'
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        if len(student_feature_maps) < self.s_layers:
            # student_feature_maps.append(
            #     torch.zeros(student_feature_maps[-1].shape, device=self.device)
            # )
            for i in range(self.s_layers - len(student_feature_maps)):
                # add a zero tensor to the student_feature_maps
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path != 'backend':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        
        student_feature_maps = torch.stack(student_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0) # (num_layers, feature_dim, batch_size, hidden_dim)
        
        # print current student_feature_maps and teacher_feature_maps shape
        # print(f"student_feature_maps shape: {student_feature_maps.shape}") # (num_layers, batch_size, feature_dim, hidden_dim) (0, 1, 2, 3)

        # reshape student_feature_maps to (num_layers, feature_dim, batch_size, hidden_dim)
        student_feature_maps = student_feature_maps.permute(0, 2, 1, 3)
        # print(f"after student_feature_maps shape: {student_feature_maps.shape}") # (num_layers, batch_size, feature_dim, hidden_dim) (0, 2, 1, 3)
        # print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # import sys
        # sys.exit()

        # Layer normalization and weighted average
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Transpose to put layer dimension at the end for LayerNorm
        # student_feature_maps: (num_layers, feature_dim, batch_size, hidden_dim) -> (feature_dim, batch_size, hidden_dim, num_layers)
        student_feature_maps_transposed = student_feature_maps.permute(1, 2, 3, 0)
        teacher_feature_maps_transposed = teacher_feature_maps.permute(1, 2, 3, 0)
        
        # Apply LayerNorm and weights
        student_feature_maps_norm = self.s_layer_norm(student_feature_maps_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_feature_maps_norm = self.t_layer_norm(teacher_feature_maps_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Transpose back to original shape
        student_feature_maps = student_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps_norm.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Average over the number of layers
        student_feature_maps = student_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.mean(dim=0)  # (feature_dim, batch_size, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
     
        
        student_feature_maps = student_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"student_feature_maps shape: {student_feature_maps.shape}")
        teacher_feature_maps = teacher_feature_maps.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_maps shape: {teacher_feature_maps.shape}")
        
        # Projection here
           
        teacher_feature_proj, teacher_feature_recon = self.t_projection(teacher_feature_maps)
        #teacher_feature_proj = teacher_feature_proj.transpose(0, 1) # (feature_dim, batch_size, hidden_dim) -> (batch_size, feature_dim, hidden_dim)
        #print(f"teacher_feature_proj shape: {teacher_feature_proj.shape}")
        
        # import sys
        # sys.exit()
        # projection_loss = self.mse_loss(teacher_feature_proj, teacher_feature_maps)
        mse_loss = self.mse_loss(student_feature_maps, teacher_feature_proj)
        cosine_loss = self.cosine_loss(student_feature_maps.contiguous().view(self.size, -1), teacher_feature_proj.contiguous().view(self.size, -1), target=torch.ones(self.size, device=student_feature_maps.device))
        recon_loss = self.mse_loss(teacher_feature_maps, teacher_feature_recon)
        #kl_loss = self.kl_loss(student_feature_maps, teacher_feature_proj)
        # hard code for weight
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        scaled_recon_loss = recon_loss * self.recon_loss_weight
        
        loss = scaled_mse_loss + scaled_cosine_loss + scaled_recon_loss
        #loss = mse_loss
        #loss = kl_loss
        
        return loss, mse_loss, cosine_loss, recon_loss

class MLP(nn.Module):
    """
    3-layer MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, device=device),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class StandardMidLoss_v6(nn.Module):
    """
    A loss module for Knowledge Distillation from two embedding features.
    Uses an MLP to project student's output to teacher's embedding dimension,
    and applies MSE + Cosine losses to match projected student and teacher outputs.
    
    Input format: (batch_size, num_layers, feature_dim, hidden_dim)
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # Layer normalization for weighted averaging across layers
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        
        self.device = device
        self.s_layers = s_layers
        
        # Projection from student dim (768) to teacher dim (1024)
        self.s_projection = MLP(input_dim=768, hidden_dim=1024, output_dim=1024, device=device)
        
        # Loss weights
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)

    def forward(self, student_feature_maps, teacher_feature_maps, **kwargs):
        """
        Args:
            student_feature_maps: (batch_size, num_layers, feature_dim, hidden_dim=768)
            teacher_feature_maps: (batch_size, num_layers, feature_dim, hidden_dim=1024)
        """
        batch_size = kwargs.get('size', student_feature_maps.shape[0])
        
        # Input shapes: (batch_size, num_layers, feature_dim, hidden_dim)
        #print(f"Input student shape: {student_feature_maps.shape}")
        #print(f"Input teacher shape: {teacher_feature_maps.shape}")
        
        # Move layer dimension to the end for LayerNorm: (batch_size, feature_dim, hidden_dim, num_layers)
        student_transposed = student_feature_maps.permute(0, 2, 3, 1)  
        teacher_transposed = teacher_feature_maps.permute(0, 2, 3, 1)   
        
        # Apply layer normalization and learnable weights
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)  # (s_layers,)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)  # (t_layers,)
        
        # Normalize and weight: (batch_size, feature_dim, hidden_dim, num_layers)
        student_normalized = self.s_layer_norm(student_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_normalized = self.t_layer_norm(teacher_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Average over layers: (batch_size, feature_dim, hidden_dim)
        student_averaged = student_normalized.mean(dim=-1)  
        teacher_averaged = teacher_normalized.mean(dim=-1)  
        
        #print(f"After averaging - Student: {student_averaged.shape}, Teacher: {teacher_averaged.shape}")
        
        # Project student features to match teacher dimension
        # student_averaged: (batch_size, feature_dim, 768) -> (batch_size, feature_dim, 1024)
        student_projected = self.s_projection(student_averaged)
        
        #print(f"After projection - Student: {student_projected.shape}")
        
        # Compute losses
        mse_loss = self.mse_loss(student_projected, teacher_averaged)
        
        # Flatten for cosine similarity: (batch_size, feature_dim * hidden_dim)
        student_flat = student_projected.view(batch_size, -1)
        teacher_flat = teacher_averaged.view(batch_size, -1)
        cosine_target = torch.ones(batch_size, device=self.device)
        
        cosine_loss = self.cosine_loss(student_flat, teacher_flat, cosine_target)
        
        # Scale and combine losses
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        
        total_loss = scaled_mse_loss + scaled_cosine_loss
        
        return total_loss, mse_loss, cosine_loss


class StandardMidLoss_v7(nn.Module):
    """
    A loss module for Knowledge Distillation from two embedding features.
    Uses an MLP to project teacher's output to student's embedding dimension,
    and applies MSE + Cosine losses to match projected teacher and teacher outputs.
    
    Input format: (batch_size, num_layers, feature_dim, hidden_dim)
    """

    def __init__(self, t_layers, s_layers, device, **kwargs):
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # Layer normalization for weighted averaging across layers
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        self.t_projection = ShallowAutoencoder(input_dim=1024, latent_dim=768, use_bias=True).to(device)
        self.device = device
        self.s_layers = s_layers
        
        # Projection from student dim (768) to teacher dim (1024)
        #self.s_projection = MLP(input_dim=768, hidden_dim=1024, output_dim=1024, device=device)
        
        # Loss weights
        self.mse_loss_weight = kwargs.get('mse_loss_weight', 0.0001)
        self.cosine_loss_weight = kwargs.get('cosine_loss_weight', 1)
        self.recon_loss_weight = kwargs.get('recon_loss_weight', 0.0001)

    def forward(self, student_feature_maps, teacher_feature_maps, **kwargs):
        """
        Args:
            student_feature_maps: (batch_size, num_layers, feature_dim, hidden_dim=768)
            teacher_feature_maps: (batch_size, num_layers, feature_dim, hidden_dim=1024)
        """
        batch_size = kwargs.get('size', student_feature_maps.shape[0])
        
        # Move layer dimension to the end for LayerNorm: (batch_size, feature_dim, hidden_dim, num_layers)
        student_transposed = student_feature_maps.permute(0, 2, 3, 1)  
        teacher_transposed = teacher_feature_maps.permute(0, 2, 3, 1)   
        
        # Apply layer normalization and learnable weights
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)  # (s_layers,)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)  # (t_layers,)
        
        # Normalize and weight: (batch_size, feature_dim, hidden_dim, num_layers)
        student_normalized = self.s_layer_norm(student_transposed) * s_norm_w.view(1, 1, 1, -1)
        teacher_normalized = self.t_layer_norm(teacher_transposed) * t_norm_w.view(1, 1, 1, -1)
        
        # Average over layers: (batch_size, feature_dim, hidden_dim)
        student_averaged = student_normalized.mean(dim=-1)  
        teacher_averaged = teacher_normalized.mean(dim=-1)  
        
        #print(f"After averaging - Student: {student_averaged.shape}, Teacher: {teacher_averaged.shape}")
        
        # Project student features to match teacher dimension
        # student_averaged: (batch_size, feature_dim, 768) -> (batch_size, feature_dim, 1024)
        #student_projected = self.s_projection(student_averaged)
        teacher_projected, teacher_recon = self.t_projection(teacher_averaged)
        
        #print(f"After projection - Student: {student_projected.shape}")
        
        # Compute losses
        mse_loss = self.mse_loss(student_averaged, teacher_projected)
        recon_loss = self.mse_loss(teacher_averaged, teacher_recon)
        
        # Flatten for cosine similarity: (batch_size, feature_dim * hidden_dim)
        student_flat = student_averaged.view(batch_size, -1)
        teacher_flat = teacher_projected.view(batch_size, -1)
        cosine_target = torch.ones(batch_size, device=self.device)
        
        cosine_loss = self.cosine_loss(student_flat, teacher_flat, cosine_target)
        
        # Scale and combine losses
        scaled_mse_loss = mse_loss * self.mse_loss_weight
        scaled_cosine_loss = cosine_loss * self.cosine_loss_weight
        scaled_recon_loss = recon_loss * self.recon_loss_weight
        
        total_loss = scaled_mse_loss + scaled_cosine_loss + scaled_recon_loss
        
        return total_loss, mse_loss, cosine_loss, recon_loss