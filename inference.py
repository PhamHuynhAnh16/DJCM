import os
import sys
import torch

import numpy as np

from scipy.signal import medfilt

sys.path.append(os.getcwd())

from src.model import DJCM
from src.spec import Spectrogram
from src.constants import SAMPLE_RATE, N_CLASS, CONST, WINDOW_LENGTH

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

    def __init__(
        self, 
        model_path: str, 
        device: str | torch.device = "cpu", 
        is_half: bool = False, 
        # onnx = False, 
        providers = ["CPUExecutionProvider"], 
        batch_size = 1, 
        hop_length = 320,
        segment_len = 5.12,
        in_channels = 1, 
        n_blocks = 1, 
        latent_layers = 1,
        kernel_size = None,
    ):
        super(DJCM_Inference, self).__init__()
        self.onnx = model_path.endswith(".onnx")

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = DJCM(in_channels or 1, n_blocks or 1, latent_layers or 1)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            self.model = model.half() if is_half else model.float()

        self.batch_size = batch_size
        self.hop_length = hop_length
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = int(self.seg_len // hop_length)

        self.device = device
        self.is_half = is_half
        self.kernel_size = kernel_size

        self.spec_extractor = Spectrogram(hop_length, WINDOW_LENGTH).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + CONST
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def infer_from_audio(self, audio, thred=0.03):
        """
        Infers F0 from audio.

        Args:
            audio (np.ndarray): Audio signal.
            thred (float, optional): Threshold for salience. Defaults to 0.03.
        """

        if torch.is_tensor(audio): audio = audio.cpu().numpy()
        if audio.ndim > 1: audio = audio.squeeze()

        with torch.inference_mode(): # with torch.no_grad():
            padded_audio = self.pad_audio(audio)
            hidden = self.inference(padded_audio)[:(audio.shape[-1] // self.hop_length + 1)]

            f0 = self.decode(hidden.squeeze(0).cpu().numpy(), thred)

            if self.kernel_size is not None:
                f0 = medfilt(f0, kernel_size=self.kernel_size)

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

    def inference(self, segments):
        hidden_segments = torch.cat([
            self.audio2hidden(self.spec_extractor(segments[i:i + self.batch_size])) # [:, :-1, :]
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