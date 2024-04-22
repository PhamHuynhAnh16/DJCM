import os

import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import shutil

df_info = pd.read_csv(r'D:\ICASSP_2024\SVSDT\dataset\MIR1K\info.csv')
path_in = r'D:\联合模型\Data\MIR-1K\Wavfile'
path_label_in = r'D:\联合模型\Data\MIR-1K\PitchLabel'
path_out = r'D:\ICASSP_2024\SVSDT\dataset\MIR1K'

for _, row in tqdm(df_info.iterrows()):
    filename, _, split = row[0], row[1], row[2]
    audio_m, _ = librosa.load(os.path.join(path_in, filename), sr=16000, mono=True)
    audio_t, _ = librosa.load(os.path.join(path_in, filename), sr=16000, mono=False)
    audio_v = audio_t[1]
    sf.write(os.path.join(path_out, split, filename.replace('.wav', '_m.wav')), audio_m.T, 16000, 'PCM_24')
    sf.write(os.path.join(path_out, split, filename.replace('.wav', '_v.wav')), audio_v.T, 16000, 'PCM_24')
    shutil.copy(os.path.join(path_label_in, filename.replace('.wav', '.pv')),
                os.path.join(path_out, split, filename.replace('.wav', '.pv')))

