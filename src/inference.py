import numpy as np
import torch


class Inference:
    def __init__(self, model, seg_len, seg_frames, hop_length, batch_size, device):
        super(Inference, self).__init__()
        self.model = model.eval()
        self.seg_len = seg_len
        self.seg_frames = seg_frames
        self.batch_size = batch_size
        self.hop_length = hop_length
        self.device = device

    def inference(self, audio):
        with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            segments = self.en_frame(padded_audio)
            sep_segments, pitch_segments = self.forward_in_mini_batch(self.model, segments)
            out_audio = self.de_frame(sep_segments, type_seg='audio')[:, :audio.shape[-1]]
            pitch_pred = self.de_frame(pitch_segments, type_seg='pitch')[:(audio.shape[-1]//self.hop_length+1)]
            return out_audio, pitch_pred

    def pad_audio(self, audio):
        c, audio_len = audio.shape
        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = seg_nums * self.seg_len - audio_len + self.seg_len // 2
        padded_audio = torch.cat([torch.zeros(c, self.seg_len // 4).to(self.device), audio,
                                  torch.zeros(c, pad_len - self.seg_len // 4).to(self.device)], dim=1)
        return padded_audio

    def en_frame(self, audio):
        c, audio_len = audio.shape
        assert audio_len % (self.seg_len // 2) == 0

        segments = []
        start = 0
        while start + self.seg_len <= audio_len:
            segments.append(audio[:, start:start + self.seg_len])
            start += self.seg_len // 2
        segments = torch.stack(segments, dim=0)
        return segments

    def forward_in_mini_batch(self, model, segments):
        out_segments = []
        pitch_segments = []
        segments_num = segments.shape[0]
        # print(segments_num, end='\t')
        batch_start = 0
        while True:
            # print('#', end='\t')
            if batch_start + self.batch_size >= segments_num:
                batch_tmp = segments[batch_start:].shape[0]
                segment_in = torch.cat([segments[batch_start:],
                                        torch.zeros_like(segments)[:self.batch_size-batch_tmp].to(self.device)], dim=0)
                # out_audio = model(segment_in)
                out_audio, pitch_pred = model(segment_in)
                out_segments.append(out_audio[:batch_tmp, :])
                pitch_segments.append(pitch_pred[:batch_tmp, :])
                break
            else:
                segment_in = segments[batch_start:batch_start+self.batch_size]
                out_audio, pitch_pred = model(segment_in)
                out_segments.append(out_audio)
                pitch_segments.append(pitch_pred)
            batch_start += self.batch_size
        out_segments = torch.cat(out_segments, dim=0)
        pitch_segments = torch.cat(pitch_segments, dim=0)

        return out_segments, pitch_segments

    def de_frame(self, segments, type_seg='audio'):
        output = []
        if type_seg == 'audio':
            for segment in segments:
                output.append(segment[:, self.seg_len // 4: int(self.seg_len * 0.75)])
            output = torch.cat(output, dim=1)
        else:
            for segment in segments:
                output.append(segment[self.seg_frames // 4: int(self.seg_frames * 0.75)])
            output = torch.cat(output, dim=0)
        return output
