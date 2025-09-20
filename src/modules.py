import torch

import torch.nn as nn
import torch.nn.functional as F

from .seq import BiGRU
from .constants import WINDOW_LENGTH, N_CLASS

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""

    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None: layer.bias.data.fill_(0.0)

def init_bn(bn):
    """Initialize a Batchnorm layer."""

    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)

class ResConvBlock(nn.Module):
    """
    A convolutional block with residual connection.

    Args:
        in_planes (int): Number of input planes.
        out_planes (int): Number of output planes.
    """

    def __init__(self, in_planes, out_planes):
        super(ResConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.is_shortcut = False

        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, (1, 1))
            self.is_shortcut = True

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut: init_layer(self.shortcut)

    def forward(self, x):
        out = self.conv2(self.act2(self.bn2(self.conv1(self.act1(self.bn1(x))))))

        if self.is_shortcut: return self.shortcut(x) + out
        else: return out + x

class ResEncoderBlock(nn.Module):
    """
    A residual encoder block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_blocks (int): Number of convolutional blocks in the block.
        kernel_size (tuple): Size of the average pooling kernel.
    """

    def __init__(self, in_channels, out_channels, n_blocks, kernel_size):
        super(ResEncoderBlock, self).__init__()
        self.conv = nn.ModuleList([ResConvBlock(in_channels, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))

        self.pool = nn.MaxPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for each_layer in self.conv:
            x = each_layer(x)

        if self.pool is not None: return x, self.pool(x)
        return x

class ResDecoderBlock(nn.Module):
    """
    A residual decoder block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_blocks (int): Number of convolutional blocks in the block.
        stride (tuple): Stride for transposed convolution.
    """

    def __init__(self, in_channels, out_channels, n_blocks, stride):
        super(ResDecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride, (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = nn.ModuleList([ResConvBlock(out_channels * 2, out_channels)])

        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x, concat):
        x = self.conv1(F.relu_(self.bn1(x)))
        x = torch.cat((x, concat), dim=1)
    
        for each_layer in self.conv:
            x = each_layer(x)
    
        return x
    
class Encoder(nn.Module):
    """
    The encoder part.

    Args:
        in_channels (int): Number of input channels.
        n_blocks (int): Number of convolutional blocks in each encoder block.
    """

    def __init__(self, in_channels, n_blocks):
        super(Encoder, self).__init__()
        self.en_blocks = nn.ModuleList([
            ResEncoderBlock(in_channels, 32, n_blocks, (1, 2)), 
            ResEncoderBlock(32, 64, n_blocks, (1, 2)), 
            ResEncoderBlock(64, 128, n_blocks, (1, 2)), 
            ResEncoderBlock(128, 256, n_blocks, (1, 2)), 
            ResEncoderBlock(256, 384, n_blocks, (1, 2)), 
            ResEncoderBlock(384, 384, n_blocks, (1, 2))
        ])

    def forward(self, x):
        concat_tensors = []

        for layer in self.en_blocks:
            _, x = layer(x)
            concat_tensors.append(_)

        return x, concat_tensors

class Decoder(nn.Module):
    """
    The decoder part.

    Args:
        n_blocks (int): Number of convolutional blocks in each encoder block.
    """

    def __init__(self, n_blocks):
        super(Decoder, self).__init__()
        self.de_blocks = nn.ModuleList([
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 384, n_blocks, (1, 2)), 
            ResDecoderBlock(384, 256, n_blocks, (1, 2)), 
            ResDecoderBlock(256, 128, n_blocks, (1, 2)), 
            ResDecoderBlock(128, 64, n_blocks, (1, 2)), 
            ResDecoderBlock(64, 32, n_blocks, (1, 2))
        ])

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1 - i])

        return x
    
class LatentBlocks(nn.Module):
    """
    Latent feature processing block consisting of repeated ResEncoderBlocks.

    Args:
        n_blocks (int): Number of sub-blocks in each ResEncoderBlock.
        latent_layers (int): Number of repeated ResEncoderBlocks in the latent space.
    """

    def __init__(self, n_blocks, latent_layers):
        super(LatentBlocks, self).__init__()
        self.latent_blocks = nn.ModuleList([
            ResEncoderBlock(384, 384, n_blocks, None) 
            for _ in range(latent_layers)
        ])

    def forward(self, x):
        for layer in self.latent_blocks:
            x = layer(x)

        return x
    
class SVS_Decoder(nn.Module):
    """
    Decoder module for SVS models.

    Args:
        in_channels (int):Number of input channels.
        n_blocks (int): Number of sub-blocks used in each decoder block.
    """

    def __init__(self, in_channels, n_blocks):
        super(SVS_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, in_channels * 4, (1, 1))
        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        return self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))

class PE_Decoder(nn.Module):
    """
    Decoder module for Pitch Estimation tasks.

    Args:
        n_blocks (int): Number of sub-blocks used in each decoder block.
        seq_layers (int, optional): Number of GRU layers.
    """

    def __init__(self, n_blocks, seq_layers=1):
        super(PE_Decoder, self).__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, 1, (1, 1))
        self.fc = nn.Sequential(BiGRU((1, WINDOW_LENGTH // 2), 1, seq_layers), nn.Linear(WINDOW_LENGTH // 2, N_CLASS), nn.Sigmoid())
        init_layer(self.after_conv2)

    def forward(self, x, concat_tensors):
        return self.fc(self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))).squeeze(1)