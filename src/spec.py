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

class ISTFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """
        PyTorch implementation of ISTFT with Conv1d. The function has the 
        same output as librosa.istft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set to False to finetune all parameters.
        """
        super(ISTFT, self).__init__()
        assert pad_mode in ['constant', 'reflect']
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = self.n_fft
        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        # Initialize Conv1d modules for calculating real and imag part of DFT.
        self.init_real_imag_conv()
        # Initialize overlap add window for reconstruct time domain signals.
        self.init_overlap_add_window()
        
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)

        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def init_real_imag_conv(self):
        """
        Initialize Conv1d for calculating real and imag part of DFT.
        """
        self.W = self.idft_matrix(self.n_fft) / self.n_fft
        self.conv_real = nn.Conv1d(in_channels=self.n_fft, out_channels=self.n_fft, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=self.n_fft, out_channels=self.n_fft, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        ifft_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        # Pad the window to n_fft
        ifft_window = librosa.util.pad_center(ifft_window, size=self.n_fft)

        self.conv_real.weight.data = torch.Tensor(np.real(self.W * ifft_window[None, :]).T)[:, :, None]
        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W * ifft_window[None, :]).T)[:, :, None]

    def init_overlap_add_window(self):
        """
        Initialize overlap add window for reconstruct time domain signals.
        """
        ola_window = librosa.filters.get_window(self.window, self.win_length, fftbins=True)
        ola_window = librosa.util.normalize(ola_window, norm=None) ** 2
        ola_window = librosa.util.pad_center(ola_window, size=self.n_fft)
        ola_window = torch.Tensor(ola_window)

        self.register_buffer('ola_window', ola_window)

    def forward(self, real_stft, imag_stft, length):
        """
        Calculate inverse STFT.

        Args:
            real_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            imag_stft: (batch_size, channels=1, time_steps, n_fft // 2 + 1)
            length: int
        
        Returns:
            real: (batch_size, data_length), output signals.
        """
        assert real_stft.ndimension() == 4 and imag_stft.ndimension() == 4
        _, _, frames_num, _ = real_stft.shape

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)

        # Get full stft representation from spectrum using symmetry attribute.
        full_real_stft, full_imag_stft = self._get_full_stft(real_stft, imag_stft)
        # Calculate IDFT frame by frame.
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)

        # Overlap add signals in frames to reconstruct signals.
        y = self._overlap_add_divide_window_sum(s_real, frames_num)
        y = self._trim_edges(y, length)
        return y

    def _get_full_stft(self, real_stft, imag_stft):
        """
        Get full stft representation from spectrum using symmetry attribute.

        Args:
            real_stft: (batch_size, n_fft // 2 + 1, time_steps)
            imag_stft: (batch_size, n_fft // 2 + 1, time_steps)

        Returns:
            full_real_stft: (batch_size, n_fft, time_steps)
            full_imag_stft: (batch_size, n_fft, time_steps)
        """
        full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)

        return full_real_stft, full_imag_stft

    def _overlap_add_divide_window_sum(self, s_real, frames_num):
        r"""Overlap add signals in frames to reconstruct signals.

        Args:
            s_real: (batch_size, n_fft, time_steps), signals in frames
            frames_num: int

        Returns:
            y: (batch_size, audio_samples)
        """
        
        output_samples = (s_real.shape[-1] - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(input=s_real, output_size=(1, output_samples), kernel_size=(1, self.win_length), stride=(1, self.hop_length))
        y = y[:, 0, 0, :]
        # (batch_size, audio_samples)

        # Get overlap-add window sum to be divided.
        ifft_window_sum = self._get_ifft_window(frames_num)
        ifft_window_sum = torch.clamp(ifft_window_sum, 1e-11, np.inf)
        y = y / ifft_window_sum[None, :]

        return y

    def _get_ifft_window(self, frames_num):
        """
        Get overlap-add window sum to be divided.

        Args:
            frames_num: int

        Returns:
            ifft_window_sum: (audio_samlpes,), overlap-add window sum to be 
            divided.
        """
        
        output_samples = (frames_num - 1) * self.hop_length + self.win_length
        window_matrix = self.ola_window[None, :, None].repeat(1, 1, frames_num)

        ifft_window_sum = F.fold(input=window_matrix, output_size=(1, output_samples), kernel_size=(1, self.win_length), stride=(1, self.hop_length))
        ifft_window_sum = ifft_window_sum.squeeze()

        return ifft_window_sum

    def _trim_edges(self, y, length):
        """
        Trim audio.

        Args:
            y: (audio_samples,)
            length: int

        Returns:
            (trimmed_audio_samples,)
        """
        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            start = self.n_fft // 2 if self.center else 0

            y = y[:, start : start + length]

        return y

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
        # stft_out = torch.stft(
        #     audio,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     win_length=self.win_length,
        #     window=self.window,
        #     center=center,
        #     pad_mode=pad_mode,
        #     return_complex=True
        # )

        # stft_out = stft_out.transpose(1, 2)
        # mag = torch.clamp(stft_out.abs(), self.clamp, float("inf"))

        # time_steps, freq_bins = mag.shape[1], mag.shape[2]
        # mag = mag.reshape(bs, c, time_steps, freq_bins)

        # return mag

class Spec2Wav(nn.Module):
    def __init__(self, hop_length, window_size):
        super(Spec2Wav, self).__init__()
        self.istft = ISTFT(window_size, hop_length, window_size)

    def magphase(self, real, imag):
        """
        Calculate magnitude and phase from real and imag part of signals.

        Args:
            real: tensor, real part of signals
            imag: tensor, imag part of signals

        Returns:
            mag: tensor, magnitude of signals
            cos: tensor, cosine of phases of signals
            sin: tensor, sine of phases of signals
        """
        mag = (real ** 2 + imag ** 2) ** 0.5
        cos = real / torch.clamp(mag, 1e-10, np.inf)
        sin = imag / torch.clamp(mag, 1e-10, np.inf)

        return mag, cos, sin

    def forward(self, x, spec_m, cos_m, sin_m, audio_len):
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)

        mask_spec = torch.sigmoid(x[:, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, 2, :, :])

        _, mask_cos, mask_sin = self.magphase(_mask_real, _mask_imag)
        linear_spec = x[:, :, 3, :, :]

        out_cos = cos_m * mask_cos - sin_m * mask_sin
        out_sin = sin_m * mask_cos + cos_m * mask_sin

        out_spec = F.relu(spec_m.detach() * mask_spec + linear_spec)
        out_real = (out_spec * out_cos).reshape(bs * c // 4, 1, time_steps, freqs_steps)

        out_imag = (out_spec * out_sin).reshape(bs * c // 4, 1, time_steps, freqs_steps)
        audio = self.istft(out_real, out_imag, audio_len).reshape(bs, c // 4, audio_len)

        return audio, out_spec