import os
import torch

model_path = r"G:\Assets\DJCM-Model\model-8280.pt"
output_path = "djcm.pt"
fp16_model = True
# use_orig_infer = False

model = torch.load(model_path, map_location="cpu", weights_only=False)

new_state_dict = {}
for k, v in model.state_dict().items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

# if not use_orig_infer:
#     del new_state_dict["to_wav.istft.ola_window"], new_state_dict["to_wav.istft.conv_real.weight"], new_state_dict["to_wav.istft.conv_imag.weight"]
#     del new_state_dict["to_spec.stft.conv_real.weight"], new_state_dict["to_spec.stft.conv_imag.weight"]

torch.save(new_state_dict, output_path)

if fp16_model:
    fp16_new_state_dict = {}

    for key in new_state_dict:
        fp16_new_state_dict[key] = new_state_dict[key].half()
    
    torch.save(fp16_new_state_dict, os.path.splitext(output_path)[0] + "_fp16.pt")