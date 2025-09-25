import torch.nn.functional as F
from torch import nn
import torch
from autoencoders import ShallowAutoencoder, DeepAutoencoder, SequentialDeepAutoencoder
from typing import Dict, Any, Optional, Union, List, Tuple

class MLP(nn.Module):
    """
    3-layer MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, device=device),
            nn.GELU()
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x


class UnifiedMidLoss(nn.Module):
    """
    A unified loss module for Middle loss for Knowledge Distillation that can dynamically handle:
    - Different projection layers (Linear, ShallowAutoencoder, DeepAutoencoder, MLP, None)
    - Different loss combinations (MSE, Cosine, Reconstruction)
    - Flexible weight configuration
    - Different input formats and processing modes
    
    Supported projection types:
    - 'linear': nn.Linear layer
    - 'shallow_ae': ShallowAutoencoder 
    - 'deep_ae': DeepAutoencoder
    - 'mlp': 3-layer MLP
    - None: No projection
    
    Supported loss types:
    - 'mse': Mean Squared Error
    - 'l1': L1 Loss (Mean Absolute Error)
    - 'cosine': Cosine Embedding Loss
    - 'recon': Reconstruction Loss (when using autoencoders)
    """
    
    def __init__(self, t_layers: int, s_layers: int, device: torch.device, 
                 projection_config: Optional[Dict[str, Any]] = None,
                 loss_config: Optional[Dict[str, Any]] = None,
                 pooling_config: Optional[Dict[str, Any]] = None,
                 processing_mode: str = 'legacy',
                 **kwargs):
        """
        Args:
            t_layers: Number of teacher layers
            s_layers: Number of student layers  
            device: Device to use
            projection_config: Configuration for projection layers
                - type: 'linear', 'shallow_ae', 'deep_ae', 'mlp', or None
                - target: 'teacher' or 'student' (which to project)
                - input_dim: Input dimension
                - output_dim: Output dimension
                - kwargs: Additional arguments for projection layer
            loss_config: Configuration for loss components
                - enabled: List of enabled loss types ['mse', 'l1', 'cosine', 'recon']
                - weights: Dict with loss weights {loss_type: weight}
            pooling_config: Configuration for pooling across layers
                - method: 'mean', 'sum', 'max', 'min', 'weighted_sum', 'last', 'first'
            processing_mode: 'legacy' (old format) or 'new' (new format)
        """
        super().__init__()
        
        self.device = device
        self.s_layers = s_layers
        self.processing_mode = processing_mode
        
        # Default configurations
        default_projection = {
            'type': None,
            'target': 'teacher',
            'input_dim': 1024,
            'output_dim': 768,
            'kwargs': {}
        }
        
        default_loss = {
            'enabled': ['mse', 'cosine'],
            'weights': {'mse': 0.0001, 'cosine': 1.0, 'recon': 0.0001, 'l1': 0.0001}
        }
        
        default_pooling = {
            'method': 'mean'
        }
        
        # Merge with provided configs
        self.projection_config = {**default_projection, **(projection_config or {})}
        self.loss_config = {**default_loss, **(loss_config or {})}
        self.pooling_config = {**default_pooling, **(pooling_config or {})}
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.l1_loss = nn.L1Loss()
        
        # Layer normalization and weights
        self.t_layer_norm = nn.LayerNorm(normalized_shape=t_layers, device=device)
        self.t_weight_hidd = nn.Parameter(torch.ones(t_layers, device=device))
        self.s_layer_norm = nn.LayerNorm(normalized_shape=s_layers, device=device)
        self.s_weight_hidd = nn.Parameter(torch.ones(s_layers, device=device))
        
        self.frontend_type = kwargs.get('frontend_type', 'fairseq')
        # Initialize projection layer
        self.projection_layer = self._create_projection_layer()
        self.AUTO_ENCODER_TYPES = ['shallow_ae', 'deep_ae', 'sequential_deep_ae']
        
    def _create_projection_layer(self) -> Optional[nn.Module]:
        """Create projection layer based on configuration"""
        proj_type = self.projection_config['type']
        
        if proj_type is None:
            return None
            
        input_dim = self.projection_config['input_dim']
        output_dim = self.projection_config['output_dim']
        kwargs = self.projection_config['kwargs']
        
        if proj_type == 'linear':
            return nn.Linear(input_dim, output_dim, device=self.device)
        
        elif proj_type == 'shallow_ae':
            return ShallowAutoencoder(
                input_dim=input_dim, 
                latent_dim=output_dim, 
                use_bias=kwargs.get('use_bias', True)
            ).to(self.device)
        
        elif proj_type == 'deep_ae':
            dims = kwargs.get('dims', [input_dim, output_dim, output_dim])
            return DeepAutoencoder(
                dims=dims,
                use_bias=kwargs.get('use_bias', True)
            ).to(self.device)
        elif proj_type == 'sequential_deep_ae':
            return SequentialDeepAutoencoder(
                dims=kwargs.get('dims', [input_dim, output_dim, output_dim]),
                use_bias=kwargs.get('use_bias', True)
            ).to(self.device)
        elif proj_type == 'mlp':
            hidden_dim = kwargs.get('hidden_dim', max(input_dim, output_dim))
            return MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
    
    def _process_legacy_input(self, student_io_dict, teacher_io_dict, 
                             student_module_path_list, student_module_io_list,
                             teacher_module_path_list, teacher_module_io_list, **kwargs):
        """Process input in legacy format (old method)"""
        size = kwargs.get('size', 64)
        student_feature_maps = []
        teacher_feature_maps = []
        
        student_conv_feature_maps = None
        teacher_conv_feature_maps = None
        # Process student features
        for student_module_path, student_module_io in zip(student_module_path_list, student_module_io_list):
            if student_module_path == 'front_end.model.encoder.pos_conv':
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                student_conv_feature_maps = student_io_dict[student_module_path][student_module_io]
            elif student_module_path != 'backend' and student_module_path != 'front_end.model.encoder.pos_conv':
                if student_module_io == 'False:True':
                    student_module_io = 'output'
                elif student_module_io == 'True:False':
                    student_module_io = 'input'
                try:
                    student_feature_maps.append(student_io_dict[student_module_path][student_module_io])
                except:
                    continue
        # Check if any feat in feats map that return batch size 1 like (feature_dim, 1, hidden_dim)
        
        # then remove it 
        student_feature_maps = [feat for feat in student_feature_maps if feat.shape[1] != 1]

        # Pad student features if needed
        if len(student_feature_maps) < self.s_layers:
            for i in range(self.s_layers - len(student_feature_maps)):
                student_feature_maps.append(
                    torch.zeros(student_feature_maps[-1].shape, device=self.device)
                )
        
        # Process teacher features
        for teacher_module_path, teacher_module_io in zip(teacher_module_path_list, teacher_module_io_list):
            if teacher_module_path == 'front_end.model.encoder.pos_conv':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_conv_feature_maps = teacher_io_dict[teacher_module_path][teacher_module_io]
            elif teacher_module_path != 'backend' and teacher_module_path != 'front_end.model.encoder.pos_conv':
                if teacher_module_io == 'False:True':
                    teacher_module_io = 'output'
                elif teacher_module_io == 'True:False':
                    teacher_module_io = 'input'
                teacher_feature_maps.append(teacher_io_dict[teacher_module_path][teacher_module_io])
        #import pdb; pdb.set_trace()
        # Stack and reshape
        student_feature_maps = torch.stack(student_feature_maps, dim=0)  # (num_layers, batch_size, feature_dim, hidden_dim)
        teacher_feature_maps = torch.stack(teacher_feature_maps, dim=0)  # (num_layers, batch_size, feature_dim, hidden_dim)
        
        if self.frontend_type == 'HF':
            student_feature_maps = student_feature_maps.permute(0, 2, 1, 3)  # (num_layers, feature_dim, batch_size, hidden_dim)
        # Handle different reshaping modes
        if hasattr(self, '_reshape_student') and self._reshape_student:
            student_feature_maps = student_feature_maps.permute(0, 2, 1, 3)  # (num_layers, feature_dim, batch_size, hidden_dim)
        
        return student_feature_maps, teacher_feature_maps, size, student_conv_feature_maps, teacher_conv_feature_maps
    
    def _apply_layer_normalization_and_pooling(self, student_feature_maps, teacher_feature_maps):
        """Apply layer normalization and pooling across layers"""
        # Layer normalization and weighted pooling
        s_norm_w = F.softmax(self.s_weight_hidd, dim=-1)
        t_norm_w = F.softmax(self.t_weight_hidd, dim=-1)
        
        # Handle different tensor shapes based on processing mode
        if self.processing_mode == 'new':
            # New format: (batch_size, num_layers, feature_dim, hidden_dim)
            student_transposed = student_feature_maps.permute(0, 2, 3, 1)  # (batch_size, feature_dim, hidden_dim, num_layers)
            teacher_transposed = teacher_feature_maps.permute(0, 2, 3, 1)  # (batch_size, feature_dim, hidden_dim, num_layers)
            
            # Apply LayerNorm and weights
            student_normalized = self.s_layer_norm(student_transposed) * s_norm_w.view(1, 1, 1, -1)
            teacher_normalized = self.t_layer_norm(teacher_transposed) * t_norm_w.view(1, 1, 1, -1)
             # Apply weight to student_transposed and teacher_transposed (it might redundant with LayerNorm)
            # student_normalized = student_transposed * s_norm_w.view(1, 1, 1, -1)
            # teacher_normalized = teacher_transposed * t_norm_w.view(1, 1, 1, -1)
            
            # Pool over layers: (batch_size, feature_dim, hidden_dim)
            student_pooled = self._apply_pooling(student_normalized, dim=-1, tensor_type='student')
            teacher_pooled = self._apply_pooling(teacher_normalized, dim=-1, tensor_type='teacher')
            
        else:
            # Legacy format: (num_layers, feature_dim, batch_size, hidden_dim)
            student_transposed = student_feature_maps.permute(1, 2, 3, 0)  # (feature_dim, batch_size, hidden_dim, num_layers)
            teacher_transposed = teacher_feature_maps.permute(1, 2, 3, 0)  # (feature_dim, batch_size, hidden_dim, num_layers)
            
            # Apply LayerNorm and weights
            student_normalized = self.s_layer_norm(student_transposed) * s_norm_w.view(1, 1, 1, -1)
            teacher_normalized = self.t_layer_norm(teacher_transposed) * t_norm_w.view(1, 1, 1, -1)
            
            # Apply weight to student_transposed and teacher_transposed (it might redundant with LayerNorm)
            # student_normalized = student_transposed * s_norm_w.view(1, 1, 1, -1)
            # teacher_normalized = teacher_transposed * t_norm_w.view(1, 1, 1, -1)
            
            # Transpose back and pool
            student_feature_maps = student_normalized.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
            teacher_feature_maps = teacher_normalized.permute(3, 0, 1, 2)  # (num_layers, feature_dim, batch_size, hidden_dim)
            
            # print(student_feature_maps.shape, teacher_feature_maps.shape)
            # import pdb; pdb.set_trace()
            # Pool over layers
            student_pooled = self._apply_pooling(student_feature_maps, dim=0, tensor_type='student')  # (feature_dim, batch_size, hidden_dim)
            teacher_pooled = self._apply_pooling(teacher_feature_maps, dim=0, tensor_type='teacher')  # (feature_dim, batch_size, hidden_dim)
            
            # Transpose to (batch_size, feature_dim, hidden_dim)
            student_pooled = student_pooled.transpose(0, 1)
            teacher_pooled = teacher_pooled.transpose(0, 1)
        
        return student_pooled, teacher_pooled
    
    def _apply_pooling(self, tensor, dim, tensor_type='student'):
        """Apply pooling operation based on configuration"""
        pooling_method = self.pooling_config.get('method', 'mean')
        
        if pooling_method == 'mean':
            return tensor.mean(dim=dim)
        elif pooling_method == 'sum':
            return tensor.sum(dim=dim)
        elif pooling_method == 'max':
            return tensor.max(dim=dim)[0]  # max returns (values, indices), we want values
        elif pooling_method == 'min':
            return tensor.min(dim=dim)[0]  # min returns (values, indices), we want values
        elif pooling_method == 'weighted_sum':
            # For weighted sum, use appropriate layer weights based on tensor type
            if tensor_type == 'student':
                weights = F.softmax(self.s_weight_hidd, dim=-1)
            else:  # teacher
                weights = F.softmax(self.t_weight_hidd, dim=-1)
            
            if dim == -1:  # New format
                weights = weights.view(1, 1, 1, -1)
            else:  # Legacy format (dim=0)
                weights = weights.view(-1, 1, 1, 1)
            return (tensor * weights).sum(dim=dim)
        elif pooling_method == 'weighted_mean':
            # For weighted mean, use appropriate layer weights based on tensor type
            if tensor_type == 'student':
                weights = F.softmax(self.s_weight_hidd, dim=-1)
            else:  # teacher
                weights = F.softmax(self.t_weight_hidd, dim=-1)
            
            if dim == -1:  # New format
                weights = weights.view(1, 1, 1, -1)
            else:  # Legacy format (dim=0)
                weights = weights.view(-1, 1, 1, 1)
            return (tensor * weights).mean(dim=dim)
        elif pooling_method == 'last':
            # Take the last layer
            if dim == -1:
                return tensor[..., -1]
            else:
                return tensor[-1]
        elif pooling_method == 'first':
            # Take the first layer
            if dim == -1:
                return tensor[..., 0]
            else:
                return tensor[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
    
    def _apply_projection(self, student_features, teacher_features):
        """Apply projection layer based on configuration"""
        proj_target = self.projection_config['target']
        proj_type = self.projection_config['type']
        
        if self.projection_layer is None:
            return student_features, teacher_features, None
        
        # Apply projection based on target
        if proj_target == 'teacher':
            if proj_type in self.AUTO_ENCODER_TYPES:
                teacher_projected, teacher_recon = self.projection_layer(teacher_features)
                return student_features, teacher_projected, teacher_recon
            else:
                teacher_projected = self.projection_layer(teacher_features)
                return student_features, teacher_projected, None
        
        elif proj_target == 'student':
            if proj_type in self.AUTO_ENCODER_TYPES:
                student_projected, student_recon = self.projection_layer(student_features)
                return student_projected, teacher_features, student_recon
            else:
                student_projected = self.projection_layer(student_features)
                return student_projected, teacher_features, None
        
        else:
            raise ValueError(f"Unknown projection target: {proj_target}")
    
    def _compute_losses(self, student_features, teacher_features, raw_student_features, raw_teacher_features, reconstructed_features, batch_size, **kwargs):
        """Compute all enabled losses"""
        losses = {}
        student_conv_feature_maps = kwargs.get('student_conv_feature_maps', None)
        teacher_conv_feature_maps = kwargs.get('teacher_conv_feature_maps', None)
        
        if 'mse' in self.loss_config['enabled']:
            losses['mse'] = self.mse_loss(student_features, teacher_features)
            if student_conv_feature_maps is not None and teacher_conv_feature_maps is not None:
                losses['mse_conv'] = self.mse_loss(student_conv_feature_maps, teacher_conv_feature_maps)
        
        if 'l1' in self.loss_config['enabled']:
            losses['l1'] = self.l1_loss(student_features, teacher_features)
            if student_conv_feature_maps is not None and teacher_conv_feature_maps is not None:
                losses['l1_conv'] = self.l1_loss(student_conv_feature_maps, teacher_conv_feature_maps)
        
        if 'cosine' in self.loss_config['enabled']:
            student_flat = student_features.contiguous().view(batch_size, -1)
            teacher_flat = teacher_features.contiguous().view(batch_size, -1)
            cosine_target = torch.ones(batch_size, device=self.device)
            losses['cosine'] = self.cosine_loss(student_flat, teacher_flat, cosine_target)
            if student_conv_feature_maps is not None and teacher_conv_feature_maps is not None:
                losses['cosine_conv'] = self.cosine_loss(student_conv_feature_maps.contiguous().view(batch_size, -1), teacher_conv_feature_maps.contiguous().view(batch_size, -1), cosine_target)
        
        if 'recon' in self.loss_config['enabled'] and reconstructed_features is not None:
            # Determine which features to reconstruct
            if self.projection_config['target'] == 'teacher':
                losses['recon'] = self.mse_loss(raw_teacher_features, reconstructed_features)
                if student_conv_feature_maps is not None and teacher_conv_feature_maps is not None:
                    losses['recon_conv'] = self.mse_loss(raw_teacher_features, reconstructed_features)
            else:
                losses['recon'] = self.mse_loss(raw_student_features, reconstructed_features)
                if student_conv_feature_maps is not None and teacher_conv_feature_maps is not None:
                    losses['recon_conv'] = self.mse_loss(raw_student_features, reconstructed_features)
        
        return losses
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that handles both legacy and new input formats
        
        Legacy format:
        forward(student_io_dict, teacher_io_dict, student_module_path_list, 
                student_module_io_list, teacher_module_path_list, teacher_module_io_list, **kwargs)
        
        New format:
        forward(student_feature_maps, teacher_feature_maps, **kwargs)
        """
        
        if self.processing_mode == 'legacy':
            # Legacy format
            student_feature_maps, teacher_feature_maps, batch_size, student_conv_feature_maps, teacher_conv_feature_maps = self._process_legacy_input(*args, **kwargs)
        else:
            # New format
            student_feature_maps, teacher_feature_maps = args[0], args[1]
            batch_size = kwargs.get('size', student_feature_maps.shape[0])
        
        # Apply layer normalization and pooling
        student_features, teacher_features = self._apply_layer_normalization_and_pooling(
            student_feature_maps, teacher_feature_maps
        )
        
        # Apply projection
        student_proj, teacher_proj, reconstructed = self._apply_projection(student_features, teacher_features)
        #import pdb; pdb.set_trace()
        
        # Compute losses
        if self.processing_mode == 'legacy':
            if student_conv_feature_maps is not None and teacher_conv_feature_maps is not None:
                student_conv_feature_maps = student_conv_feature_maps.permute(0, 2, 1)
                teacher_conv_feature_maps = teacher_conv_feature_maps.permute(0, 2, 1)
                student_conv_feature_proj, teacher_conv_feature_proj, reconstructed_conv = self._apply_projection(student_conv_feature_maps, teacher_conv_feature_maps)
            else:
                student_conv_feature_proj = None
                teacher_conv_feature_proj = None
                reconstructed_conv = None
            losses = self._compute_losses(student_proj, teacher_proj, student_features, teacher_features, reconstructed, batch_size, student_conv_feature_maps=student_conv_feature_proj, teacher_conv_feature_maps=teacher_conv_feature_proj)
        else:
            losses = self._compute_losses(student_proj, teacher_proj, student_features, teacher_features, reconstructed, batch_size)
        
        # Scale and combine losses
        total_loss = 0
        scaled_losses = {}
        
        for loss_name, loss_value in losses.items():
            weight = self.loss_config['weights'].get(loss_name, 1.0)
            scaled_loss = loss_value * weight
            scaled_losses[loss_name] = scaled_loss
            total_loss += scaled_loss
        
        # Return format similar to original classes
        # Build return tuple dynamically based on enabled losses
        return_values = [total_loss]
        
        # Add individual losses in a consistent order
        for loss_type in ['mse', 'l1', 'cosine', 'recon']:
            if loss_type in losses:
                return_values.append(losses[loss_type])
        
        return tuple(return_values)