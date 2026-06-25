# import torch
# import torch.nn as nn
# from torch.nn.modules.transformer import _get_clones
# from .conformer import ConformerBlock

# def sinusoidal_embedding(n_channels, dim):
#     pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
#                             for p in range(n_channels)])
#     pe[:, 0::2] = torch.sin(pe[:, 0::2])
#     pe[:, 1::2] = torch.cos(pe[:, 1::2])
#     return pe.unsqueeze(0)

# class MyConformer(nn.Module):
#     def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1, pooling='mean', type='conv', **kwargs):
#         super(MyConformer, self).__init__()
#         self.pooling=pooling
#         self.dim_head=int(emb_size/heads)
#         self.dim=emb_size
#         self.heads=heads
#         self.kernel_size=kernel_size
#         self.n_encoders=n_encoders
#         self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
#         self.conv_dropout = kwargs.get('conv_dropout', 0.0)
#         self.ff_dropout = kwargs.get('ff_dropout', 0.0)
#         self.attn_dropout = kwargs.get('attn_dropout', 0.0)
#         self.encoder_blocks=_get_clones(ConformerBlock(dim = emb_size, dim_head=self.dim_head, heads= heads, 
#                 ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size, type=type, 
#                 conv_dropout = self.conv_dropout, ff_dropout = self.ff_dropout, attn_dropout = self.attn_dropout
#                 ),
#             n_encoders)
#         self.class_token = nn.Parameter(torch.rand(1, emb_size))
#         self.fc5 = nn.Linear(emb_size, 2)
    
#     # def forward(self, x): # x shape [bs, tiempo, frecuencia]
#     #     x = x + self.positional_emb[:, :x.size(1), :]
#     #     x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(x.shape[0])])#[bs,1+tiempo,emb_size]
#     #     list_attn_weight = []
#     #     for layer in self.encoder_blocks:
#     #         x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
#     #         list_attn_weight.append(attn_weight)
#     #     if self.pooling=='mean':
#     #         embedding = x.mean(dim=1)
#     #     elif self.pooling=='max':
#     #         embedding = x.max(dim=1)[0]
#     #     else:
#     #         # first token
#     #         embedding=x[:,0,:] #[bs, emb_size]
#     #     out=self.fc5(embedding) #[bs,2]
#     #     return out
    
#     # Inference optimized version
#     def forward(self, x): # x shape [bs, tiempo, frecuencia]
#         # Add positional embeddings efficiently
#         x = x + self.positional_emb[:, :x.size(1), :]
        
#         # More efficient way to add class token
#         batch_size = x.shape[0]
#         class_tokens = self.class_token.expand(batch_size, -1)
#         x = torch.cat([class_tokens.unsqueeze(1), x], dim=1)  # [bs, 1+tiempo, emb_size]
        
#         # Process through encoder blocks without accumulating attention weights
#         for layer in self.encoder_blocks:
#             x, _ = layer(x)  # Discard attention weights to save memory
#         embedding=x[:,0,:] #[bs, emb_size]
#         out=self.fc5(embedding) #[bs,2]
#         return out
    


import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones

def sinusoidal_embedding(n_channels, dim):
    """Create sinusoidal positional embeddings"""
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)


class MyConformer(nn.Module):
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, 
                 kernel_size=16, n_encoders=1, pooling='first', 
                 type='conv', max_seq_len=10000, **kwargs):
        super(MyConformer, self).__init__()
        self.pooling = pooling
        self.dim_head = int(emb_size / heads)
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders
        self.max_seq_len = max_seq_len
        
        # Make positional embedding non-trainable and contiguous
        pos_emb = sinusoidal_embedding(max_seq_len, emb_size)
        self.register_buffer('positional_emb', pos_emb.contiguous())
        
        self.conv_dropout = kwargs.get('conv_dropout', 0.0)
        self.ff_dropout = kwargs.get('ff_dropout', 0.0)
        self.attn_dropout = kwargs.get('attn_dropout', 0.0)
        
        # Import your ConformerBlock here
        from .conformer import ConformerBlock
        self.encoder_blocks = _get_clones(
            ConformerBlock(
                dim=emb_size, dim_head=self.dim_head, heads=heads,
                ff_mult=ffmult, conv_expansion_factor=exp_fac,
                conv_kernel_size=kernel_size, type=type,
                conv_dropout=self.conv_dropout,
                ff_dropout=self.ff_dropout,
                attn_dropout=self.attn_dropout
            ),
            n_encoders
        )

        
        # Keep original shape for checkpoint compatibility
        self.register_buffer('class_token', torch.randn(1, emb_size))
        
        self.fc5 = nn.Linear(emb_size, 2)
    
    def forward(self, x):
        """
        Optimized forward pass for mobile inference
        Args:
            x: [batch_size, seq_len, emb_size]
        Returns:
            out: [batch_size, 2]
        """
        batch_size, seq_len, emb_size = x.shape
        
        # Efficient positional embedding addition
        # Use contiguous slicing and avoid dynamic operations
        pos_emb = self.positional_emb[:, :seq_len, :].contiguous()
        x = x + pos_emb
        
        # Efficient class token prepending - optimized for [1, emb_size] shape
        # Expand class token to batch size: [1, emb_size] -> [batch_size, 1, emb_size]
        cls_tokens = self.class_token.unsqueeze(1).expand(batch_size, -1, -1)
        # Concatenate: [batch_size, 1, emb_size] + [batch_size, seq_len, emb_size]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, emb_size]
        
        # Process through encoder blocks
        # Discard attention weights to reduce memory and computation
        for layer in self.encoder_blocks:
            x, _ = layer(x)
        
        # Use first token (class token) for classification
        embedding = x[:, 0, :]  # [batch_size, emb_size]
        
        # Final classification
        out = self.fc5(embedding)  # [batch_size, 2]
        return out


class Distil_XLSR_N_Trans_Layer_ConformerTCM(nn.Module):
    def __init__(self, device, args, **kwargs):
        super().__init__()
        # Assuming My_XLSR_FE is your frontend
        # self.front_end = My_XLSR_FE(device, **kwargs).to(device)
        
        # For demo - replace with your actual frontend
        self.front_end_out_dim = kwargs.get('frontend_dim', 768)
        
        self.LL = nn.Linear(self.front_end_out_dim, args['emb_size'])
        
        # Replace BatchNorm2d with LayerNorm for better mobile performance
        self.first_norm = nn.LayerNorm(args['emb_size'])
        
        # Avoid inplace operations for better JIT compatibility
        self.selu = nn.SELU(inplace=False)
        
        self.backend = MyConformer(**args)
    
    def forward(self, x):
        """
        Optimized forward pass
        Args:
            x: Input tensor [batch_size, time, 1] or [batch_size, time]
        Returns:
            out: [batch_size, 2]
        """
        # Handle input shape efficiently
        if x.dim() == 3:
            x = x.squeeze(-1)  # [batch_size, time]
        
        # Extract features from frontend
        # x_ssl_feat = self.front_end.extract_feat(x)
        # For demo purposes:
        x_ssl_feat = x.unsqueeze(-1).expand(-1, -1, self.front_end_out_dim)
        
        # Linear projection
        x = self.LL(x_ssl_feat)  # [batch_size, seq_len, emb_size]
        
        # Normalization and activation (no reshape needed!)
        x = self.first_norm(x)
        x = self.selu(x)
        
        # Backend processing
        out = self.backend(x)
        return out