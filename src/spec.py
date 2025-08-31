import torch
import librosa

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class STFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """
        PyTorch implementation of STFT with Conv1d. The function has the 
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set to False to finetune all parameters.
        """
        super(STFT, self).__init__()
        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)
        # Pad the window out to n_fft size.
        fft_window = librosa.util.pad_center(fft_window, size=n_fft)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)
        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1, groups=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1, groups=1, bias=False)

        # Initialize Conv1d weights.
        self.conv_real.weight.data = torch.Tensor(np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """
        Calculate STFT of batch of signals.

        Args: 
            input: (batch_size, data_length), input signals.
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag
    
    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

class Spectrogram(nn.Module):
    """
    Extracts Spectrogram features from audio.

    Args:
        hop_length (int): Hop size between frames in samples.
        win_length (int): Length of the window function in samples.
        n_fft (int, optional): Length of the FFT window. Defaults to None, which uses win_length.
        clamp (float, optional): Minimum value for clamping the spectrogram. Defaults to 1e-5.
    """

    def __init__(self, hop_length, win_length, n_fft=None, clamp=1e-10):
        super(Spectrogram, self).__init__()
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.stft = STFT(self.n_fft, self.hop_length, self.win_length)
        # self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, audio, center=True):
        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)

        real, imag = self.stft(audio[:, :-1])
        mag = torch.clamp(real ** 2 + imag ** 2, self.clamp, np.inf) ** 0.5

        # cos = real / mag
        # sin = imag / mag

        _, _, time_steps, freq_bins = mag.shape

        mag = mag.reshape(bs, c, time_steps, freq_bins)
        # cos = cos.reshape(bs, c, time_steps, freq_bins)
        # sin = sin.reshape(bs, c, time_steps, freq_bins)

        return mag #, cos, sin
    
        # torch.stft causes severe quality degradation when training the model

        # fft = torch.stft(
        #     audio, 
        #     n_fft=self.n_fft, 
        #     hop_length=self.hop_length, 
        #     win_length=self.win_length, 
        #     window=self.window, 
        #     center=center, 
        #     return_complex=True
        # )

        # magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2) + 1e-8)
        # log_spec = torch.log(torch.clamp(magnitude, min=self.clamp))

        # return log_spec.unsqueeze(1).transpose(2, 3)