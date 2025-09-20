import os

import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import shutil
import numpy as np


df_info = pd.read_csv(r"F:\github\DJCM\dataset\MIR1K\info.csv")
path_in = r"F:\dataset\MIR-1K\Wavfile"
path_label_in = r"F:\dataset\MIR-1K\PitchLabel"
hop_length = 160
path_out = r"F:\dataset\dataset"
os.makedirs(path_out, exist_ok=True)

for _, row in tqdm(df_info.iterrows()):
    filename, _, split = row[0], row[1], row[2]
    audio_m, sr = librosa.load(os.path.join(path_in, filename), sr=16000, mono=True)

    os.makedirs(os.path.join(path_out, split), exist_ok=True)
    sf.write(os.path.join(path_out, split, filename.replace('.wav', '_m.wav')), audio_m.T, sr, 'PCM_24')

    pv_in = os.path.join(path_label_in, filename.replace('.wav', '.pv'))
    pv_out = os.path.join(path_out, split, filename.replace('.wav', '.pv'))

    if hop_length != 320:
        f0 = np.loadtxt(pv_in)

        hop_size = hop_length / 16000
        old_times = np.arange(len(f0)) * 0.02
        new_times = np.arange(0, old_times[-1] + hop_size, hop_size)

        f0_interp = np.interp(new_times, old_times, f0)
        f0_interp[np.isnan(f0_interp)] = 0.0

        np.savetxt(pv_out, f0_interp, fmt="%.6f")
    else:
        shutil.copy(pv_in, pv_out)