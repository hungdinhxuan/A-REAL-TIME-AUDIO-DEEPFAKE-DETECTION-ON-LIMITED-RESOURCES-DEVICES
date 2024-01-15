import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
from fairseq.models.distilXLSR import DistilXLSR, DistilXLSRConfig

# class Wav2Vec2Model(nn.Module):

#     def __init__(self, cp_path, device, model_type='base'):
#         super().__init__()
#         self.SUPPORT_LISTS = ['base', 'xlsr', 'distil-xlsr']
#         if model_type not in self.SUPPORT_LISTS:
#             raise ValueError('Unknown model_type of Wav2Vec2 model: {} it should in {}'.format(model_type, self.SUPPORT_LISTS))
#         self.model_type = model_type

#         if model_type == 'base' or model_type == 'xlsr':
#             model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
#             self.model = model[0]
#         else:
#             checkpoint = torch.load(cp_path, map_location=device)
#             self.pretrained_model_cfg = checkpoint["Config"]["model"]
#             self.pretrained_model_cfg = DistilXLSRConfig(self.pretrained_model_cfg)
#             self.model = DistilXLSR(self.pretrained_model_cfg)
#             self.model.load_state_dict(checkpoint["Student"])
        
#         self.out_dim = 768 if self.model_type == 'base' else 1024

#     def forward(self, input_data):
#         if self.model_type == 'base' or self.model_type == 'xlsr':
#             input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data 
#             emb = self.model(input_tmp, mask=False, features_only=True)['x']
#             return emb
#         else:
#             input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data        
            
#             (final_output, layer_results), padding_mask = self.model(
#                     source=input_tmp, 
#                     ret_layer_results=True
#                 )
#             if self.model.encoder.layer_norm_first:
#                 layer_hiddens = [i[2] for i in layer_results]
#                 layer_hiddens.pop(0)
#                 layer_hiddens.append(final_output)
#             else:
#                 layer_hiddens = [i[0] for i in layer_results]
                
#             x = layer_hiddens[-1]
#             return x

class SSLModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        cp_path = '/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.out_dim = 768
     
        print("Wav2Vec2 Model loaded successfully.")

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data 
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb
    
class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        cp_path = '/nfs/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.out_dim = 1024

    def extract_feat(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data 
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class DistilSSLModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        cp_path = '/nfs/datab/hungdx/KDW2V-AASISTL/distilXLSR_xlsr128.pt'   # Change the pre-trained XLSR model path. 
        checkpoint = torch.load(cp_path, map_location=device)
        self.pretrained_model_cfg = checkpoint["Config"]["model"]
        self.pretrained_model_cfg = DistilXLSRConfig(self.pretrained_model_cfg)
        self.model = DistilXLSR(self.pretrained_model_cfg)
        self.model.load_state_dict(checkpoint["Student"])
        self.out_dim = 1024

    def forward(self, input_data):

        torch.autograd.set_detect_anomaly(True)
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data        
        
        (final_output, layer_results), padding_mask = self.model(
                source=input_tmp, 
                ret_layer_results=True
            )
        if self.model.encoder.layer_norm_first:
            layer_hiddens = [i[2] for i in layer_results]
            layer_hiddens.pop(0)
            layer_hiddens.append(final_output)
        else:
            layer_hiddens = [i[0] for i in layer_results]
            
        x = layer_hiddens[-1]
        return x


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):

        super().__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

''' Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu and Nicholas Evans. 
    AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks. 
    In Proc. ICASSP 2022, pp: 6367--6371.'''


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        #print('x1',x1.shape)
        #print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        #print('num_type1',num_type1)
        #print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        #print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        #print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        #print('Concat x1 and x2',x.shape)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            #print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        #print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        #print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        #print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        #print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        #print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        #print('out',out.shape)
        out = self.conv1(x)

        #print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        #out = self.mp(out)
        return out
