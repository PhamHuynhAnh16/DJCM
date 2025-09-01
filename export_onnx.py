import os
import onnx
import types
import torch
import onnxsim
import onnxconverter_common

from src.model import DJCM
from src.spec import Spectrogram
from src.constants import WINDOW_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "djcm.pt"
output_path = "djcm.onnx"
fp16_model = True

in_channels = 1
n_blocks = 1
latent_layers = 1
hop_length = 320

spec_extractor = Spectrogram(hop_length, WINDOW_LENGTH).to(device)

model = DJCM(in_channels or 1, n_blocks or 1, latent_layers or 1)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
model = model.to(device).eval()

waveform = torch.randn(1, 1, 16384, dtype=torch.float32, device=device).clip(min=-1., max=1.)
spec = spec_extractor(waveform)

# Export the model

torch.onnx.export(
    model,
    (
        spec,
    ),
    output_path,
    do_constant_folding=True, 
    verbose=False, 
    input_names=[
        'spec',
    ],
    output_names=[
        'f0'
    ],
    dynamic_axes={
        'spec': [2],
        'f0': [1]
    },
)

model, _ = onnxsim.simplify(output_path)
onnx.save(model, output_path)

# Convert model to float16 arithmetic
if fp16_model:
    convert_model = onnxconverter_common.convert_float_to_float16(onnx.load(output_path), keep_io_types=True)
    onnx.save(convert_model, os.path.splitext(output_path)[0] + "_fp16.onnx")