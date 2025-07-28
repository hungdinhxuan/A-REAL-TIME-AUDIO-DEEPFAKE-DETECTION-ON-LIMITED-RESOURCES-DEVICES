import torch.nn as nn
from conformer_tcm.model import MyConformer
from models import SSLModel
from torchdistill.models.registry import register_model

@register_model(key="W2V2_ConformerTCM")
class Model(nn.Module):
    def __init__(self, args, device, ssl_cpkt_path, out_dim=1024):
        super().__init__()
        self.front_end = SSLModel(device, ssl_cpkt_path, out_dim)
        self.LL = nn.Linear(self.front_end.out_dim, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
 
    def forward(self, x):
        x.requires_grad = True
        x_ssl_feat = self.front_end.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out =self.backend(x)
        return out