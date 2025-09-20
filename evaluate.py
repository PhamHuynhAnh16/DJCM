import torch

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from mir_eval.melody import voicing_false_alarm, voicing_recall
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, overall_accuracy, raw_chroma_accuracy

from src.spec import Spectrogram
from src.constants import SAMPLE_RATE, WINDOW_LENGTH, CONST

class Inference:
    def __init__(self, model, seg_len, seg_frames, hop_length, batch_size, device):
        super(Inference, self).__init__()
        self.model = model.eval()
        self.seg_len = seg_len
        self.seg_frames = seg_frames
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.device = device
        self.spec_extractor = Spectrogram(hop_length, WINDOW_LENGTH).to(device)

    def inference(self, audio):
        with torch.no_grad():
            if torch.is_tensor(audio): audio = audio.cpu().numpy()
            if audio.ndim > 1: audio = audio.squeeze()

            padded_audio = self.pad_audio(audio)
            hidden = self.infer(padded_audio)[:(audio.shape[-1] // self.hop_length + 1)]

            return hidden

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

    def infer(self, segments):
        hidden_segments = torch.cat([
            self.model(self.spec_extractor(segments[i:i + self.batch_size])) # [:, :-1, :] 
            for i in range(0, len(segments), self.batch_size)
        ], dim=0)

        hidden = torch.cat([
            seg[self.seg_frames // 4: int(self.seg_frames * 0.75)]
            for seg in hidden_segments
        ], dim=0)

        return hidden
    
def to_local_average_cents(salience, thred=0.0):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        to_local_average_cents.cents_mapping = (np.linspace(0, 7180, 360) + CONST)

    if salience.ndim == 1:
        center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)

        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)

        return product_sum / weight_sum if np.max(salience) > thred else 0
    
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], thred) for i in range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")

def evaluate(dataset, model, batch_size, hop_length, seq_l, device, pitch_th=0.5):
    metrics = defaultdict(list)
    seq_l = int(seq_l * SAMPLE_RATE)

    seg_frames = seq_l // hop_length
    infer = Inference(model, seq_l, seg_frames, hop_length, batch_size, device)

    for data in tqdm(dataset):
        audio_m = data['audio_m'].to(device)
        pitch_label = data['pitch'].to(device)

        pitch_pred = infer.inference(audio_m)
        loss_pitch = F.binary_cross_entropy(pitch_pred, pitch_label)

        metrics['loss_pe'].append(loss_pitch.item())

        cents = to_local_average_cents(pitch_label.detach().cpu().numpy(), pitch_th)
        cents_pred = to_local_average_cents(pitch_pred.detach().cpu().numpy(), pitch_th)
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

        print(rpa, '\t', rca)

    return metrics