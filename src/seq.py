from torch import nn
from einops.layers.torch import Rearrange


class BiGRU(nn.Module):
    def __init__(self, image_size, patch_size, channels, depth):
        super(BiGRU, self).__init__()
        image_width, image_height = pair(image_size)
        patch_width, patch_height = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=patch_width, p2=patch_height),
        )
        self.gru = nn.GRU(patch_dim, patch_dim // 2, num_layers=depth, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        return self.gru(x)[0]


class BiLSTM(nn.Module):
    def __init__(self, image_size, patch_size, channels, depth):
        super(BiLSTM, self).__init__()
        image_width, image_height = pair(image_size)
        patch_width, patch_height = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)', p1=patch_width, p2=patch_height),
        )
        self.lstm = nn.LSTM(patch_dim, patch_dim // 2, num_layers=depth, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)