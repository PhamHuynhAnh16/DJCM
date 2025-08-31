import torch

import torch.nn as nn
import torch.nn.functional as F

from .constants import WINDOW_LENGTH
from .modules import Encoder, LatentBlocks, SVS_Decoder, PE_Decoder, init_bn

class DJCM(nn.Module):
    def __init__(self, in_channels, n_blocks, latent_layers):
        super(DJCM, self).__init__()
        # self.to_spec = Wav2Spec(int(hop_length / 1000 * SAMPLE_RATE), WINDOW_LENGTH)
        self.bn = nn.BatchNorm2d(WINDOW_LENGTH // 2 + 1, momentum=0.01)
        self.svs_encoder = Encoder(in_channels, n_blocks)
        self.svs_latent = LatentBlocks(n_blocks, latent_layers)
        self.svs_decoder = SVS_Decoder(in_channels, n_blocks)
        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks)
        init_bn(self.bn)

    def spec(self, x, spec_m):
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
        mask_spec = torch.sigmoid(x[:, :, 0, :, :])
        linear_spec = x[:, :, 3, :, :]

        out_spec = F.relu(spec_m.detach() * mask_spec + linear_spec)
        return out_spec

    def forward(self, spec):
        # Use Spectrogram outside the model instead of in the model
        # spec = self.to_spec(audio)

        x, concat_tensors = self.svs_encoder(self.bn(spec.transpose(1, 3)).transpose(1, 3)[..., :-1])
        x = self.svs_decoder(self.svs_latent(x), concat_tensors)
    
        out_spec = self.spec(F.pad(x, pad=(0, 1)), spec)[..., :-1]
        x, concat_tensors = self.pe_encoder(out_spec)
        pe_out = self.pe_decoder(self.pe_latent(x), concat_tensors)

        return pe_out