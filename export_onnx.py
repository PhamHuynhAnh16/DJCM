import os
import onnx
import types
import torch
import onnxsim
import onnxconverter_common

from inference import DJCM

model_path = "djcm.pt"
output_path = "djcm.onnx"
fp16_model = True

# ONNX does not support ComplexFloat

def forward(self, audio):
    # Replaced torchlibrosa module with torch.stft.

    bs, c, segment_samples = audio.shape
    audio = audio.reshape(bs * c, segment_samples)
    stft_out = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_size, window=self.window, return_complex=False, center=True, pad_mode='reflect')

    # mag = torch.abs(stft_out).permute(0, 2, 1)
    mag = torch.sqrt(stft_out[..., 0]**2 + stft_out[..., 1]**2 + 1e-9).permute(0, 2, 1)
    mag = mag.reshape(bs, c, mag.shape[1], mag.shape[2])

    return mag

model = DJCM(1, 1, 10, 1, 16000 // 10)
# Replace forward in to_spec
model.to_spec.forward = types.MethodType(forward, model.to_spec)

model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model = model.to("cpu").eval()

waveform = torch.randn(1, 1, 16384, dtype=torch.float32, device="cpu").clip(min=-1., max=1.)

# Export the model

torch.onnx.export(
    model,
    (
        waveform,
    ),
    output_path,
    do_constant_folding=True, 
    verbose=False, 
    input_names=[
        'audio',
    ],
    output_names=[
        'f0'
    ],
    dynamic_axes={
        'audio': [2],
        'f0': [1]
    },
)

model, _ = onnxsim.simplify(output_path)
onnx.save(model, output_path)

# Convert model to float16 arithmetic
if fp16_model:
    convert_model = onnxconverter_common.convert_float_to_float16(onnx.load(output_path), keep_io_types=True)
    onnx.save(convert_model, os.path.splitext(output_path)[0] + "_fp16.onnx")