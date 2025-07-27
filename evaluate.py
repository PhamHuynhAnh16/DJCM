import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from collections import defaultdict
import soundfile as sf
import torch.nn.functional as F
from src import to_local_average_cents, Inference
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, overall_accuracy, raw_chroma_accuracy
from mir_eval.melody import voicing_false_alarm, voicing_recall
from src import SAMPLE_RATE


def calculate_sdr(ref, est):
    s_true = ref
    s_artif = est - ref
    sdr = 10.0 * (
        torch.log10(torch.clip(torch.mean(s_true ** 2), 1e-8))
        - torch.log10(torch.clip(torch.mean(s_artif ** 2), 1e-8))
    )
    return sdr


def evaluate(dataset, model, batch_size, hop_length, seq_l, device, path=None, pitch_th=0.5):
    metrics = defaultdict(list)
    seq_l = int(seq_l * SAMPLE_RATE)
    hop_length = int(hop_length / 1000 * SAMPLE_RATE)
    seg_frames = seq_l // hop_length
    infer = Inference(model, seq_l, seg_frames, hop_length, batch_size, device)

    for data in tqdm(dataset):
        audio_m = data['audio_m'].to(device)
        audio_v = data['audio_v'].to(device)
        pitch_label = data['pitch'].to(device)

        audio_v_pred, pitch_pred = infer.inference(audio_m)
        loss_svs = F.l1_loss(audio_v_pred, audio_v)
        loss_pitch = F.binary_cross_entropy(pitch_pred, pitch_label)
        loss = loss_svs + loss_pitch
        metrics['loss_svs'].append(loss_svs.item())
        metrics['loss_pe'].append(loss_pitch.item())
        metrics['loss_total'].append(loss.item())

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

        if path is not None:
            sf.write(os.path.join(path, data['file'].replace('_v.wav', '.wav')), audio_v_pred.cpu().numpy(),
                     samplerate=16000)
            df_pitch = pd.DataFrame(columns=['times', 'freqs', 'confi'])
            df_pitch['times'] = time_slice
            df_pitch['freqs'] = freqs_pred
            df_pitch['confi'] = torch.max(pitch_pred, dim=-1).values.numpy()
            df_pitch.to_csv(os.path.join(path, data['file'].replace('_v.wav', '.csv')), index=False)
        sdr = calculate_sdr(audio_v, audio_v_pred).item()
        sdr1 = calculate_sdr(audio_v, audio_m).item()
        metrics['SDR'].append(sdr)
        metrics['NSDR'].append(sdr - sdr1)
        metrics['NSDR_W'].append(len(audio_v) * (sdr - sdr1))
        metrics['LENGTH'].append(len(audio_v))
        print(sdr, '\t', rpa, '\t', rca)

    return metrics
