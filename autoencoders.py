import math
from abc import abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Sequence, Union, Tuple



# set device globally
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AbstractAutoencoder(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ShallowAutoencoder(AbstractAutoencoder):
    """ Standard shallow AE with 1 fully-connected layer in the encoder and 1 in the decoder """
    def __init__(self, input_dim: int = 784, latent_dim: int = 200, use_bias=True):
        super().__init__()
        assert input_dim > 0 and latent_dim > 0
        self.type = "shallowAE"
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim, bias=use_bias), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim, bias=use_bias), nn.Sigmoid())


class DeepAutoencoder(AbstractAutoencoder):
    """ Standard deep AE """
    def __init__(self, dims: Sequence[int], use_bias=True):
        """
        :param dims: seq of integers specifying the dimensions of the layers (length of dims = number of layers)
        :param use_bias: if False, don't use bias
        """
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        self.type = "deepAE"
        self.use_bias = use_bias
        enc_layers = []
        dec_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            enc_layers.append(nn.ReLU(inplace=True))
        for i in reversed(range(1, len(dims))):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1], bias=use_bias))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()
        self.encoder = nn.Sequential(nn.Flatten(), *enc_layers)
        self.decoder = nn.Sequential(*dec_layers)


class SequentialDeepAutoencoder(AbstractAutoencoder):
    """ Deep AE that processes each frame independently while preserving sequence structure """
    def __init__(self, dims: Sequence[int], use_bias=True):
        """
        :param dims: seq of integers specifying the dimensions of the layers 
                    (first dim should be feature_size, last dim is the compressed size)
        :param use_bias: if False, don't use bias
        """
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        self.type = "sequentialDeepAE"
        self.use_bias = use_bias
        
        # Build encoder layers
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            if i < len(dims) - 2:  # Don't add ReLU after the last layer
                enc_layers.append(nn.ReLU(inplace=True))
        
        # Build decoder layers
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1], bias=use_bias))
            if i > 1:  # Don't add activation after the last layer
                dec_layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_size)
        batch_size, seq_len, feature_size = x.shape
        #import pdb; pdb.set_trace()
        # Reshape to process all frames at once: (batch_size * seq_len, feature_size)
        x_reshaped = x.reshape(-1, feature_size)
        
        # Encode
        encoded_reshaped = self.encoder(x_reshaped)
        latent_dim = encoded_reshaped.shape[-1]
        
        # Reshape back to sequence format: (batch_size, seq_len, latent_dim)
        encoded = encoded_reshaped.view(batch_size, seq_len, latent_dim)
        
        # Decode
        decoded_reshaped = self.decoder(encoded_reshaped)
        decoded = decoded_reshaped.view(batch_size, seq_len, feature_size)
        
        return encoded, decoded

class DeepRandomizedAutoencoder(DeepAutoencoder):
    def __init__(self, dims: Sequence[int]):
        super().__init__(dims=dims, use_bias=False)
        self.type = "deepRandAE"

    def fit(self, num_epochs=10, bs=32, lr=0.1, momentum=0., **kwargs):
        """
        The training of this model is a pretraining of its layers where in the corresponding shallow AE
        only its decoder's weights are trained, the encoder's ones are fixed.
        Then copy the shallow decoder's weights in the corresponding layer of the bigger decoder
        and its transpose in the corresponding layer of the bigger encoder.
        """
        assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1
        self.pretrain_layers(num_epochs=num_epochs, bs=bs, lr=lr, momentum=momentum, freeze_enc=True)


class ShallowConvAutoencoder(AbstractAutoencoder):
    """ Convolutional AE with 1 conv layer in the encoder and 1 in the decoder """
    def __init__(self, channels=1, n_filters=10, kernel_size: int = 3, central_dim=100,
                 inp_side_len: Union[int, Tuple[int, int]] = 28):
        super().__init__()
        self.type = "shallowConvAE"
        pad = (kernel_size - 1) // 2  # pad to keep the original area after convolution
        central_side_len = math.floor(inp_side_len / 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=central_side_len ** 2 * n_filters, out_features=central_dim),
            nn.ReLU(inplace=True))

        # set kernel size, padding and stride to get the correct output shape
        kersize = 2 if central_side_len * 2 == inp_side_len else 3
        self.decoder = nn.Sequential(
            nn.Linear(in_features=central_dim, out_features=central_side_len ** 2 * n_filters),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(n_filters, central_side_len, central_side_len)),
            nn.ConvTranspose2d(in_channels=n_filters, out_channels=channels, kernel_size=kersize, stride=2, padding=0),
            nn.Sigmoid())


class DeepConvAutoencoder(AbstractAutoencoder):
    """ Conv Ae with variable number of conv layers """
    def __init__(self, inp_side_len=28, dims: Sequence[int] = (5, 10),
                 kernel_sizes: int = 3, central_dim=100, pool=True):
        super().__init__()
        self.type = "deepConvAE"

        # initial checks
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(dims)
        assert len(kernel_sizes) == len(dims) and all(size > 0 for size in kernel_sizes)

        # build encoder
        step_pool = 1 if len(dims) < 3 else (2 if len(dims) < 6 else 3)
        side_len = inp_side_len
        side_lengths = [side_len]
        dims = (1, *dims)
        enc_layers = []
        for i in range(len(dims) - 1):
            pad = (kernel_sizes[i] - 1) // 2
            enc_layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=kernel_sizes[i],
                                        padding=pad, stride=1))
            enc_layers.append(nn.ReLU(inplace=True))
            if pool and (i % step_pool == 0 or i == len(dims) - 1) and side_len > 3:
                enc_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
                side_len = math.floor(side_len / 2)
                side_lengths.append(side_len)

        # fully connected layers in the center of the autoencoder to reduce dimensionality
        fc_dims = (side_len ** 2 * dims[-1], side_len ** 2 * dims[-1] // 2, central_dim)
        self.encoder = nn.Sequential(
            *enc_layers,
            nn.Flatten(),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(inplace=True)
        )

        # build decoder
        central_side_len = side_lengths.pop(-1)
        # side_lengths = side_lengths[:-1]
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            # set kernel size, padding and stride to get the correct output shape
            kersize = 2 if len(side_lengths) > 0 and side_len * 2 == side_lengths.pop(-1) else 3
            pad, stride = (1, 1) if side_len == inp_side_len else (0, 2)
            # create transpose convolution layer
            dec_layers.append(nn.ConvTranspose2d(in_channels=dims[i], out_channels=dims[i - 1], kernel_size=kersize,
                                                 padding=pad, stride=stride))
            side_len = side_len if pad == 1 else (side_len * 2 if kersize == 2 else side_len * 2 + 1)
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Sigmoid()

        self.decoder = nn.Sequential(
            nn.Linear(fc_dims[2], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(dims[-1], central_side_len, central_side_len)),
            *dec_layers,
        )
