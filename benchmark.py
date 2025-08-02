import torch

import numpy as np

from tqdm import tqdm
from collections import defaultdict
from mir_eval.melody import voicing_false_alarm, voicing_recall
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, overall_accuracy, raw_chroma_accuracy

from src.model import DJCM
from src.dataset import MIR1K
from src.inference import Inference
from src.constants import SAMPLE_RATE
from src.utils import to_local_average_cents

def evaluate(dataset, model, batch_size, hop_length, seq_l, device, pitch_th=0.5):
    metrics = defaultdict(list)
    seq_l = int(seq_l * SAMPLE_RATE)
    hop_length = int(hop_length / 1000 * SAMPLE_RATE)
    seg_frames = seq_l // hop_length
    infer = Inference(model, seq_l, seg_frames, hop_length, batch_size, device)

    for data in tqdm(dataset):
        audio_m = data['audio_m'].to(device)
        pitch_label = data['pitch'].to(device)

        with torch.inference_mode():
            _, pitch_pred = infer.inference(audio_m)

        cents = to_local_average_cents(pitch_label.detach().cpu().numpy(), None, pitch_th)
        cents_pred = to_local_average_cents(pitch_pred.detach().cpu().numpy(), None, pitch_th)

        freqs = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents])
        freqs_pred = np.array([10 * (2 ** (cent / 1200)) if cent else 0 for cent in cents_pred])

        time_slice = np.array([i * hop_length / SAMPLE_RATE for i in range(len(freqs))])
        ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freqs, time_slice, freqs_pred)

        rpa = raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
        rca = raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
        oa = overall_accuracy(ref_v, ref_c, est_v, est_c)
        vfa = voicing_false_alarm(ref_v, est_v)
        vr = voicing_recall(ref_v, est_v)

        metrics['RPA'].append(rpa)
        metrics['RCA'].append(rca)
        metrics['OA'].append(oa)
        metrics['VFA'].append(vfa)
        metrics['VR'].append(vr)

        print("\n", rpa, '\t', rca, '\t', oa)

    return metrics

if __name__ == "__main__":
    pitch_th = 0.03
    hop_length = 20
    batch_size = 8
    seq_l = 2.56
    seq_frames = int(seq_l * 1000 / hop_length)

    in_channels = 1
    n_blocks = 1
    latent_layers = 1

    model_path = "djcm.pt"
    dataset_path = "dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DJCM(in_channels, n_blocks, hop_length, latent_layers, seq_frames)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True), strict=False)
    model = model.to(device).eval()

    valid_dataset = MIR1K(path=dataset_path, hop_length=hop_length, groups=['test'], sequence_length=None)
    metrics = evaluate(valid_dataset, model, batch_size, hop_length, seq_l, device, pitch_th=pitch_th)

    rpa = np.round(np.mean(metrics['RPA']) * 100, 2)
    rca = np.round(np.mean(metrics['RCA']) * 100, 2)
    oa = np.round(np.mean(metrics['OA']) * 100, 2)

    print(f"RPA: {rpa} RCA: {rca} OA: {oa}")