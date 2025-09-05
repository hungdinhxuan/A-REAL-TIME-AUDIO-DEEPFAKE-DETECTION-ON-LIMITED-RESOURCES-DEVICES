import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os

from models import SSLModel
import torch.nn.functional as F
from torchdistill.models.registry import register_model


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
# FOR fine-tuned SSL MODEL
############################


class DropoutForMC(nn.Module):
    """Dropout layer for Bayesian model
    THe difference is that we do dropout even in eval stage
    """

    def __init__(self, p, dropout_flag=True):
        super(DropoutForMC, self).__init__()
        self.p = p
        self.flag = dropout_flag
        return

    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, training=self.flag)


class BackEnd(nn.Module):
    """Back End Wrapper
    """

    def __init__(self, input_dim, out_dim, num_classes,
                 dropout_rate, dropout_flag=True):
        super(BackEnd, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes

        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag

        # a simple full-connected network for frame-level feature processing
        self.m_frame_level = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),

            nn.Linear(self.in_dim, self.in_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag),

            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate)
        )
        # DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))

        # linear layer to produce output logits
        self.m_utt_level = nn.Linear(self.out_dim, self.num_class)

        return

    def forward(self, feat):
        """ logits, emb_vec = back_end_emb(feat)

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          logits: tensor, (batch, num_output_class)
          emb_vec: tensor, (batch, emb_dim)

        """
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        feat_ = self.m_frame_level(feat)

        # average pooling -> (batch, self.out_dim)
        feat_utt = feat_.mean(1)

        # output linear
        logits = self.m_utt_level(feat_utt)
        return logits


class VIB(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, m_mcdp_rate=0.5, mcdp_flag=True):
        super(VIB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.m_mcdp_rate = m_mcdp_rate
        self.m_mcdp_flag = mcdp_flag

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
        )

        # Latent space
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            torch.nn.Dropout(self.m_mcdp_rate),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return z, decoded, mu, logvar


@register_model(key="W2V2_VIB")
class Model(nn.Module):
    def __init__(self, device, ssl_cpkt_path, out_dim=1024):
        super().__init__()
        self.device = device

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device, ssl_cpkt_path, out_dim)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.loss_CE = nn.CrossEntropyLoss()
        self.VIB = VIB(128, 128, 64)
        self.backend = BackEnd(64, 64, 2, 0.5, False)
        self.gelu = nn.GELU()

        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)
        # Post-processing

    def _forward(self, x):
        # -------pre-trained Wav2vec model fine tunning ------------------------##
        # if self.flag_fix_ssl:
        #     with torch.no_grad():
        #         x_ssl_feat = self.ssl_model.extract_feat(
        #             x.squeeze(-1), is_train=False)
        # else:
        #     x_ssl_feat = self.ssl_model.extract_feat(
        #         x.squeeze(-1), is_train=self.is_train)  # (bs,frame_number,feat_dim)
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))

        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)
        feats = x
        x = self.gelu(x)

        # VIB
        # x [batch, frame_number, 64]
        x, decoded, mu, logvar = self.VIB(x)

        # output [batch, 2]
        # emb [batch, 128]
        output = self.backend(x)
        # output = F.log_softmax(output, dim=1)
        # if (self.is_train):
        #     return output, feats, emb
        return output

    def forward(self, x_big):

        return self._forward(x_big)


# if __name__ == "__main__":
#     from torchinfo import summary
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = Model(
#         device=device,
#         ssl_cpkt_path='/datad/hungdx/Rawformer-implementation-anti-spoofing/pretrained/xlsr2_300m.pt',
#     ).to(device)
#     summary(model, input_size=(8, 64600))
