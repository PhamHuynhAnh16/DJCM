import os
import sys
import torch

import numpy as np

sys.path.append(os.getcwd())

from src.model import DJCM
from src.dataset import MIR1K
from evaluate import evaluate

if __name__ == "__main__":
    in_channels = 1
    n_blocks = 1
    latent_layers = 1

    seq_l = 2.56
    hop_length = 160
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "djcm.pt"
    pitch_th = 0.08
    batch_size = 4

    model = DJCM(in_channels or 1, n_blocks or 1, latent_layers or 1)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = model.to(device).float().eval()

    valid_dataset = MIR1K(path="dataset", hop_length=hop_length, groups=['test'], sequence_length=None)
    print('valid nums:', len(valid_dataset))

    metrics = evaluate(valid_dataset, model, batch_size, hop_length, seq_l, device, pitch_th)

    rpa = np.round(np.mean(metrics['RPA']) * 100, 2)
    rpa_std = np.round(np.std(metrics['RPA']) * 100, 2)

    rca = np.round(np.mean(metrics['RCA']) * 100, 2)
    rca_std = np.round(np.std(metrics['RCA']) * 100, 2)

    oa = np.round(np.mean(metrics['OA']) * 100, 2)
    oa_std = np.round(np.std(metrics['OA']) * 100, 2)

    print(f"RPA: {rpa}±{rpa_std} RCA: {rca}±{rca_std} OA: {oa}±{oa_std}")