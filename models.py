import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import logging
# from fairseq.models.distilXLSR import DistilXLSR, DistilXLSRConfig
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel, AutoConfig
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H, WAV2VEC2_BASE
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from transformers.models.wav2vec2.convert_wav2vec2_original_pytorch_checkpoint_to_pytorch import recursively_load_weights
from typing import Optional
from torchaudio.models import wav2vec2_model
from torchdistill.models.registry import register_model
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


class SSL_WAV2VEC2_ASR_BASE_960H_TA(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = WAV2VEC2_ASR_BASE_960H.get_model().to(device)
        self.out_dim = 768

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        features, _ = self.model.extract_features(input_tmp)
        features = features[0]
        return features


class SSL_WAV2VEC2_BASE_TA(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = WAV2VEC2_BASE.get_model().to(device)
        self.out_dim = 768

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        features, _ = self.model.extract_features(input_tmp)
        features = features[0]
        return features


class SSL_WAV2VEC2_BASE_FSTA(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Change the pre-trained XLSR model path.
        cp_path = '/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 cp_path])
        self.model = model[0]
        self.model = import_fairseq_model(self.model)
        self.model = self.model.to(device)
        self.out_dim = 768

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        features, _ = self.model.extract_features(input_tmp)
        features = features[0]
        return features


class Distil_SSL_WAV2VEC2_BASE_TAHG(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained('OthmaneJ/distil-wav2vec2')
        self.model = import_huggingface_model(self.model)
        self.model = self.model.to(device)
        self.out_dim = 768

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        features, _ = self.model.extract_features(input_tmp)
        print(features)
        features = features[0]
        return features


class SSLHuggingFaceModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", out_dim=768, device="cuda"):
        super().__init__()

        # fairseq_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(["wav2vec_small.pt"])
        # fairseq_model = fairseq_model[0]

        config = Wav2Vec2Config.from_pretrained(model_name)
        # self.model = AutoModelForPreTraining.from_pretrained(model_name, config=config)
        self.model = Wav2Vec2Model.from_pretrained(model_name, config=config)

        # Recursively load weights from fairseq model
        # recursively_load_weights(fairseq_model, self.model, is_headless=False)
        # del fairseq_model
        self.model = self.model.to(device)
        self.out_dim = out_dim

    def forward(self, input_data) -> Tensor:
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(
            input_tmp, output_hidden_states=True).hidden_states[-1]
        return emb


class SSLModelBase(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Change the pre-trained XLSR model path.
        cp_path = '/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 cp_path])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = 768
        self.freeze = False
        print("Wav2Vec2 Base Fairseq Model init")

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

    def frozen(self):
        logging.info("Freezing the model")
        for param in self.model.parameters():
            param.requires_grad = False
        self.freeze = True

    def unfrozen(self):
        logging.info("Unfreezing the model")
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze = False


class SSLModelFTBase(nn.Module):
    def __init__(self):
        super().__init__()
        # Change the pre-trained XLSR model path.
        cp_path = '/datab/hungdx/KDW2V-AASISTL/wav2vec_small_960h.pt'
        model_override_rules = {}
        model_override_rules['task'] = {'_name': 'audio_finetuning'}
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path], arg_overrides=model_override_rules)
        self.model = model[0]
        self.out_dim = 768

        print("Wav2Vec2 Model loaded successfully.")

    def forward(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


class SSLModel(nn.Module):
    def __init__(self, device, cp_path, out_dim):
        super().__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 cp_path])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = out_dim
        self.freeze = False

    def extract_feat(self, input_data):
        input_tmp = input_data[:, :, 0] if input_data.ndim == 3 else input_data
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

    def forward(self, input_data):
        return self.extract_feat(input_data)

    def frozen(self):
        logging.info("Freezing the model")
        for param in self.model.parameters():
            param.requires_grad = False
        self.freeze = True

    def unfrozen(self):
        logging.info("Unfreezing the model")
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze = False


class DistilSSLModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Change the pre-trained XLSR model path.
        cp_path = '/datab/hungdx/KDW2V-AASISTL/distilXLSR_xlsr128.pt'
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

    def forward(self, x1, x2, master: Optional[torch.Tensor] = None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        # print('num_type1',num_type1)
        # print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        # print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        # print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('Concat x1 and x2',x.shape)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            # print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        # Convert num_type1 to a tensor if it's not already
        if not isinstance(num_type1, torch.Tensor):
            num_type1 = torch.tensor(num_type1)

        # Convert num_type2 to a tensor if it's not already
        if not isinstance(num_type2, torch.Tensor):
            num_type2 = torch.tensor(num_type2)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        # print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        # print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        # print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x2',x2.shape)
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
        # self.k = k
        self.k = torch.tensor(k)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        # Convert self.k to a tensor if it's not already

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
        # n_nodes = max(int(n_nodes * k), 1)
        # n_nodes = torch.max(torch.tensor([int(n_nodes * k), 1]))
        n_nodes = torch.max(
            (torch.as_tensor(n_nodes) * k).long(), torch.as_tensor(1))
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        self.bn1 = None
        self.conv_downsample = None

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
        if not self.first and self.bn1 is not None:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        # print('out',out.shape)
        out = self.conv1(x)

        # print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)

        if self.downsample and self.conv_downsample is not None:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out


# -------------------------------------- RES2NET --------------------------------------
class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def middle_indices(array_length, number_of_middle_elements):
    # Calculate the start index
    start_index = (array_length - number_of_middle_elements) // 2
    # Calculate the end index
    end_index = start_index + number_of_middle_elements
    # Create a list of the middle indices
    middle_indices = list(range(start_index, end_index))
    return middle_indices


@register_model(key='My_XLSR_FE')
class My_XLSR_FE(nn.Module):

    def __init__(self, device, **kwargs):
        super().__init__()
        self.num_layers = kwargs.get('num_layers', 24)
        self.order = kwargs.get('order', 'first')
        self.custom_order = kwargs.get('custom_order', None)
        if self.num_layers < 1 or self.num_layers > 24:
            raise ValueError(
                "Number of layers must be at least 1 and at most 24.")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
                                                                                 '/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt'])
        self.model = model[0]
        self.model = self.model.to(device)
        self.out_dim = 1024

        if self.order == 'last':
            # Get the last n layers
            self.model.encoder.layers = self.model.encoder.layers[-self.num_layers:]
        elif self.order == 'first':
            # Get the first n layers
            self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]
        elif self.order == 'middle':
            indices = middle_indices(24, self.num_layers)

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in indices])
        else:
            if self.custom_order is None:
                raise ValueError(
                    "Custom order must be provided as a list of integers (0-23).")

            # Check if the custom order is valid
            if type(self.custom_order) != list:
                raise ValueError("Custom order must be a list of integers.")

            # if len(self.custom_order) != self.num_layers:
            #     raise ValueError(
            #         "Length of custom order must be less than or equal to the number of layers.")
            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in self.custom_order])

    def forward(self, x):
        return self.extract_feat(x)

    def extract_feat(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        emb = self.model(input_tmp, mask=False, features_only=True)[
            'x']
        return emb

    def extract_layer_results(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        layer_results = self.model(input_tmp, mask=False, features_only=True)[
            'layer_results']
        return layer_results


@register_model(key='Custom_Wav2Vec2_Fe')
class Custom_Wav2Vec2_Fe(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()

        self.out_dim = kwargs.get('out_dim', 256)
        encoder_layer_drop = kwargs.get('encoder_layer_drop', 0.0)
        encoder_dropout = kwargs.get('encoder_dropout', 0.0)
        encoder_ff_interm_dropout = kwargs.get(
            'encoder_ff_interm_dropout', 0.0)
        encoder_attention_dropout = kwargs.get(
            'encoder_attention_dropout', 0.0)
        encoder_projection_dropout = kwargs.get(
            'encoder_projection_dropout', 0.0)
        encoder_num_layers = kwargs.get('encoder_num_layers', 12)

        self.model = wav2vec2_model(
            extractor_mode="layer_norm",
            extractor_conv_bias=True,
            encoder_embed_dim=self.out_dim,
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_pos_conv_kernel=128,
            encoder_pos_conv_groups=16,
            encoder_num_layers=encoder_num_layers,  # Number of transformer layers
            encoder_num_heads=16,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_features=4096,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_first=True,
            encoder_layer_drop=encoder_layer_drop,
            extractor_conv_layer_config=None,  # Default to match the wav2vec2_xlsr_300m model
            aux_num_out=None,
        ).to(device)

    def forward(self, x):
        input_tmp = x[:, :, 0] if x.ndim == 3 else x
        feat, _ = self.model(input_tmp)
        return feat

    def extract_feat(self, x):
        return self.forward(x)


class Res2Net(nn.Module):
    def __init__(self, features_size, stride_=1, scale=4, padding_=1, groups_=1, reduction=16):
        super(Res2Net, self).__init__()
        # erro for wrong input
        if scale < 2 or features_size % scale:
            print('Error:illegal input for scale or feature size')

        self.divided_features = int(features_size / scale)
        self.conv1 = nn.Conv2d(features_size, features_size,
                               kernel_size=1, stride=stride_, padding=0, groups=groups_)
        self.conv2 = nn.Conv2d(self.divided_features, self.divided_features,
                               kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
        self.convs = nn.ModuleList()
        self.se = SEblock(features_size, reduction)
        for i in range(scale - 2):

            self.convs.append(
                nn.Conv2d(self.divided_features, self.divided_features,
                          kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
            )

    def forward(self, x):
        features_in = x
        conv1_out = self.conv1(features_in)
        y1 = conv1_out[:, 0:self.divided_features, :, :]
        fea = self.conv2(
            conv1_out[:, self.divided_features:2*self.divided_features, :, :])
        features = fea
        for i, conv in enumerate(self.convs):
            pos = (i + 1)*self.divided_features
            divided_feature = conv1_out[:, pos:pos+self.divided_features, :, :]
            fea = conv(fea + divided_feature)
            features = torch.cat([features, fea], dim=1)

        out = torch.cat([y1, features], dim=1)
        conv1_out1 = self.conv1(out)
        se_out = self.se(conv1_out1)
        result = features_in + se_out
        return result
