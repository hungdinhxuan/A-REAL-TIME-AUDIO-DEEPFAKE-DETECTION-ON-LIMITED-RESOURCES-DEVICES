import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones
from .conformer import ConformerBlock

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe.unsqueeze(0)

class MyConformer(nn.Module):
    def __init__(self, emb_size=128, heads=4, ffmult=4, exp_fac=2, kernel_size=16, n_encoders=1, pooling='mean', type='conv', **kwargs):
        super(MyConformer, self).__init__()
        self.pooling=pooling
        self.dim_head=int(emb_size/heads)
        self.dim=emb_size
        self.heads=heads
        self.kernel_size=kernel_size
        self.n_encoders=n_encoders
        self.positional_emb = nn.Parameter(sinusoidal_embedding(10000, emb_size), requires_grad=False)
        self.conv_dropout = kwargs.get('conv_dropout', 0.0)
        self.ff_dropout = kwargs.get('ff_dropout', 0.0)
        self.attn_dropout = kwargs.get('attn_dropout', 0.0)
        self.encoder_blocks=_get_clones(ConformerBlock(dim = emb_size, dim_head=self.dim_head, heads= heads, 
                ff_mult = ffmult, conv_expansion_factor = exp_fac, conv_kernel_size = kernel_size, type=type, 
                conv_dropout = self.conv_dropout, ff_dropout = self.ff_dropout, attn_dropout = self.attn_dropout
                ),
            n_encoders)
        self.class_token = nn.Parameter(torch.rand(1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)
    
    def forward(self, x): # x shape [bs, tiempo, frecuencia]
        x = x + self.positional_emb[:, :x.size(1), :]
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])#[bs,1+tiempo,emb_size]
        list_attn_weight = []
        for layer in self.encoder_blocks:
            x, attn_weight = layer(x) #[bs,1+tiempo,emb_size]
            list_attn_weight.append(attn_weight)
        if self.pooling=='mean':
            embedding = x.mean(dim=1)
        elif self.pooling=='max':
            embedding = x.max(dim=1)[0]
        else:
            # first token
            embedding=x[:,0,:] #[bs, emb_size]
        out=self.fc5(embedding) #[bs,2]
        return out
    

