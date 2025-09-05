import torch.nn.functional as F
from torch import nn
import torch
# Define a linear layer to transform from 256 to 1024 dimensions


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