import os
import sys
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import medfilt
from einops.layers.torch import Rearrange

SAMPLE_RATE, WINDOW_LENGTH, N_CLASS, HOP_SIZE = 16000, 2048, 360, 160

sys.path.append(os.getcwd())

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

class Wav2Spec(nn.Module):
    """
    Extract Spectrogram frequency from audio.

    Args:
        hop_length (int): Hop size between frames in samples.
        win_length (int): Length of the window function in samples.
    """

    def __init__(self, hop_length, window_size):
        super(Wav2Spec, self).__init__()
        self.hop_length = hop_length
        self.window_size = window_size
        self.n_fft = window_size
        self.register_buffer("window", torch.hann_window(window_size), persistent=False)

    def forward(self, audio):
        # Replaced torchlibrosa module with torch.stft.

        bs, c, segment_samples = audio.shape
        audio = audio.reshape(bs * c, segment_samples)

        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_size, window=self.window, center=True, return_complex=True, pad_mode='reflect')
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        return magnitude.unsqueeze(1).transpose(2, 3)

class DJCM(nn.Module):
    def __init__(self, in_channels, n_blocks, latent_layers):
        super(DJCM, self).__init__()
        self.bn = nn.BatchNorm2d(WINDOW_LENGTH // 2 + 1, momentum=0.01)
        self.svs_encoder = Encoder(in_channels, n_blocks)
        self.svs_latent = LatentBlocks(n_blocks, latent_layers)
        self.svs_decoder = SVS_Decoder(in_channels, n_blocks)
        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks)
        self.to_spec = Wav2Spec(HOP_SIZE, WINDOW_LENGTH)
        init_bn(self.bn)

    def spec(self, x, spec_m):
        # Spec2Wav shortcode keeps only out_spec

        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
        mask_spec = torch.sigmoid(x[:, :, 0, :, :])
        linear_spec = x[:, :, 3, :, :]

        out_spec = F.relu(spec_m.detach() * mask_spec + linear_spec)
        return out_spec

    def forward(self, audio):
        # Removed Spec2Wav as unnecessary, only kept Wav2Spec code.

        # This code is a shortcode to extract F0 (No test)
        # spec_m = self.to_spec(audio)
        # x, concat_tensors = self.pe_encoder(self.bn(spec_m.transpose(1, 3)).transpose(1, 3)[..., :-1])
        # pe_out = self.pe_decoder(self.pe_latent(x), concat_tensors)

        spec = self.to_spec(audio)
        x, concat_tensors = self.svs_encoder(self.bn(spec.transpose(1, 3)).transpose(1, 3)[..., :-1])
        x = self.svs_decoder(self.svs_latent(x), concat_tensors)
    
        out_spec = self.spec(F.pad(x, pad=(0, 1)), spec)[..., :-1]
        x, concat_tensors = self.pe_encoder(out_spec)
        pe_out = self.pe_decoder(self.pe_latent(x), concat_tensors)

        return pe_out

class DJCM_Inference:
    """
    A predictor for fundamental frequency (F0) based on the Deep Joint Cascade Model.

    Args:
        model_path (str): Path to the DJCM model file.
        device (str, torch.device, optional): Device to use for computation. Defaults to Cpu, which uses CUDA if available.
        is_half (bool, optional): Use Half to save resources and speed up.
        onnx (bool, optional): Using the ONNX model.
        providers (list, optional): Providers of onnx model. default is CPUExecutionProvider.
        batch_size (int, optional): .
        segment_len (float, optional): .
    """

    def __init__(self, model_path, device = "cpu", is_half = False, onnx = False, providers = ["CPUExecutionProvider"], batch_size = 1, segment_len = 5.12):
        super(DJCM_Inference, self).__init__()
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = DJCM(1, 1, 1)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            self.model = model.half() if is_half else model.float()

        self.batch_size = batch_size
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = int(self.seg_len // HOP_SIZE)
        self.is_half = is_half
        self.device = device

        # get this part from RMVPE
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def audio2hidden(self, audio):
        """
        Convert audio frequency to hidden representation.

        Args:
            audio (torch.Tensor): Audio features.
        """

        if self.onnx:
            hidden = torch.as_tensor(
                self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: audio.cpu().numpy().astype(np.float32)})[0], device=self.device
            )
        else:
            hidden = self.model(
                audio.half() if self.is_half else audio.float()
            )

        return hidden

    def infer_from_audio(self, audio, thred=0.03):
        """
        Infers F0 from audio.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """

        with torch.inference_mode():
            padded_audio = self.pad_audio(audio)
            hidden = self.inference(padded_audio)[:(audio.shape[-1] // HOP_SIZE + 1)]

            f0 = self.decode(hidden.squeeze(0).cpu().numpy(), thred)
            f0 = medfilt(f0, kernel_size=3)

            return f0
        
    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        """
        Infers F0 from audio with pitch.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
            f0_min (float, int, optional): Minimum F0 threshold.
            f0_max (float, int, optional): Maximum F0 threshold.
        """

        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0

        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        """
        Converts salience to local average cents.

        Args:
            salience (np.ndarray): Salience values.
            thred (float, optional): Threshold for salience. Defaults to 0.05.
        """

        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided
        
    def decode(self, hidden, thred=0.03):
        """
        Decodes hidden representation to F0.

        Args:
            hidden (np.ndarray): Hidden representation.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """

        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def pad_audio(self, audio):
        audio_len = audio.shape[-1]

        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = int(seg_nums * self.seg_len - audio_len + self.seg_len // 2)

        left_pad = np.zeros(int(self.seg_len // 4), dtype=np.float32)
        right_pad = np.zeros(int(pad_len - self.seg_len // 4), dtype=np.float32)
        padded_audio = np.concatenate([left_pad, audio, right_pad], axis=-1)

        segments = [padded_audio[start: start + int(self.seg_len)] for start in range(0, len(padded_audio) - int(self.seg_len) + 1, int(self.seg_len // 2))]
        segments = np.stack(segments, axis=0)
        segments = torch.from_numpy(segments).unsqueeze(1).to(self.device)

        return segments

    def inference(self, segments):
        hidden_segments = torch.cat([
            self.audio2hidden(segments[i:i + self.batch_size]) 
            for i in range(0, len(segments), self.batch_size)
        ], dim=0)

        hidden = torch.cat([
            seg[self.seg_frames // 4: int(self.seg_frames * 0.75)]
            for seg in hidden_segments
        ], dim=0)

        return hidden

if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    y, sr = librosa.load(r"C:\Users\Pham Huynh Anh PC\Downloads\Vocals.wav", sr=SAMPLE_RATE, mono=True)

    # https://huggingface.co/AnhP/DJCM-Test/resolve/main/djcm.pt

    model = DJCM_Inference(r"G:\Assets\DJCM-Model\djcm.pt", device="cpu")
    f0 = model.infer_from_audio(y, thred=0.03)

    with open("f0-djcm.txt", "w") as f:
        f.write(str(f0))

    print(f0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title("DJCM")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig("f0-djcm.png")
    plt.close()

    with open("f0-djcm-2.txt", "w") as f:
        for i, f0_value in enumerate(f0):
            f.write(f"{i * sr / 160},{f0_value}\n")