import torch

from torch import nn
from einops.layers.torch import Rearrange

class BiGRU(nn.Module):
    """
    A bidirectional GRU layer.

    Args:
        patch_size (tuple): Patch Height and Width Values.
        channels (int): Number of channels.
        depth (int): Number of depths.
    """

    def __init__(self, patch_size, channels, depth):
        super(BiGRU, self).__init__()
        patch_width, patch_height = patch_size
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=patch_width, p2=patch_height))
        self.gru = nn.GRU(patch_dim, patch_dim // 2, num_layers=depth, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        try:
            return self.gru(x)[0]
        except:
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]