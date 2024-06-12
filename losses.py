import torch.nn.functional as F
from torch import nn
import torch
# Define a linear layer to transform from 256 to 1024 dimensions


class ConvExpand(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(ConvExpand, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Expecting x of shape [batch_size, sequence_length, in_features]
        # Change to [batch_size, in_channels, sequence_length] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # Change back to [batch_size, sequence_length, out_features]
        x = x.permute(0, 2, 1)
        return x


class MSELoss(nn.Module):
    def __init__(self, in_features=256, out_features=1024):
        super(MSELoss, self).__init__()
        print("Adaptive MSELoss with in_features: {} and out_features: {}".format(
            in_features, out_features))
        self.expand = ConvExpand(in_features, out_features)
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.expand(x), y)


class CosineLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLoss, self).__init__()
        self.expand = ConvExpand(in_features, out_features)
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.size = out_features

    def forward(self, x, y):
        x = self.expand(x)
        x = x.view(self.size, -1)
        y = y.view(self.size, -1)
        return self.cosine_loss(x, y, torch.ones(self.size).to(x.device))
