import torch
import torch.nn as nn
from .frontend import Frontend_S, Frontend_L, Frontend_SE, Frontend_SE_W2V
from .positional_aggregator import PositionalAggregator1D
from .classifier import RawformerClassifier
from torchdistill.models.registry import register_model


class Rawformer_S(nn.Module):
    """

    """

    def __init__(self, device, transformer_hidden=64, sample_rate: int = 16000):
        super(Rawformer_S, self).__init__()
        self.front_end = Frontend_S(
            sinc_kernel_size=128, sample_rate=sample_rate)

        # this max_ft is for input of 4-sec and 16000 sample-rate
        self.positional_embedding = PositionalAggregator1D(
            max_C=64, max_ft=23*16, device=device)

        self.classifier = RawformerClassifier(
            C=64, n_encoder=2, transformer_hidden=transformer_hidden)  # output: [batch, C]

    def forward(self, x):
        x = self.front_end(x)
        x = self.positional_embedding(x)
        x = self.classifier(x)
        return x


class Rawformer_L(nn.Module):

    def __init__(self, device, transformer_hidden=80, sample_rate: int = 16000):
        super(Rawformer_L, self).__init__()
        self.front_end = Frontend_L(
            sinc_kernel_size=128, sample_rate=sample_rate)

        # this max_ft is for input of 4-sec and 16000 sample-rate
        self.positional_embedding = PositionalAggregator1D(
            max_C=64, max_ft=23*29, device=device)

        self.classifier = RawformerClassifier(
            C=64, n_encoder=3, transformer_hidden=transformer_hidden)  # output: [batch, C]

    def forward(self, x):
        x = self.front_end(x)
        x = self.positional_embedding(x)
        x = self.classifier(x)
        return x


class Rawformer_SE(nn.Module):

    def __init__(self, device, transformer_hidden=660, sample_rate: int = 16000):
        super(Rawformer_SE, self).__init__()
        self.front_end = Frontend_SE(
            sinc_kernel_size=128, sample_rate=sample_rate)

        # this max_ft is for input of 4-sec and 16000 sample-rate
        self.positional_embedding = PositionalAggregator1D(
            max_C=64, max_ft=23*16, device=device)

        self.classifier = RawformerClassifier(
            C=64, n_encoder=2, transformer_hidden=transformer_hidden)  # output: [batch, C]

    def forward(self, x):
        x = self.front_end(x)
        x = self.positional_embedding(x)
        x = self.classifier(x)
        return x


@register_model(key="Rawformer_SE_W2V")
class Rawformer_SE_W2V(nn.Module):

    def __init__(self, device, ssl_cpkt_path, transformer_hidden=660):
        super(Rawformer_SE_W2V, self).__init__()
        self.front_end = Frontend_SE_W2V(device, ssl_cpkt_path, 1024)

        # this max_ft is for input of 4-sec and 16000 sample-rate
        self.positional_embedding = PositionalAggregator1D(
            max_C=64, max_ft=66*16, device=device)

        self.classifier = RawformerClassifier(
            C=64, n_encoder=3, transformer_hidden=transformer_hidden)  # output: [batch, C]

    def forward(self, x):
        x = self.front_end(x)
        x = self.positional_embedding(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = Rawformer_SE_W2V(
        "cuda:2", "/nfs/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt", 1024).to("cuda:2")

    summary(model, (2, 16000*4))
