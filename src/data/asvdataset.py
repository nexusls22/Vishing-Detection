# Dataset class (path where data is found, subset for right protocol to dataset, processor for computing, sample rate has to 16KHz, number of max samples)
import os

import soundfile as sf
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

class ASVDataset(Dataset):
    def __init__(self, data_root: str, subset: str, processor: Wav2Vec2Processor, target_sr: int = 16000, max_samples: int = None):
        self.data_root = data_root
        self.subset = subset
        self.processor = processor
        self.target_sr = target_sr

        # Map path to right protocol for sample set
        protocol_path = os.path.join(self.data_root, 'protocols', 'ASVspoof2019_LA_cm_protocols', {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt',
        }[subset])

        self.df = pd.read_csv(protocol_path, sep=' ', header=None, names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label'])

        if max_samples:
            self.df = self._stratified_sample(max_samples)

        self.audio_dir = os.path.join(self.data_root , 'raw', f'ASVspoof2019_LA_{subset}', 'flac')

        print(f'DEBUG: audio path: {self.audio_dir}')


    # Stratifying samples for even distribution of both labels
    def _stratified_sample(self, n_samples: int):
        df_bonafide = self.df[self.df['label'] == 'bonafide']
        df_spoof = self.df[self.df['label'] == 'spoof']
        n_per_class = n_samples // 2

        # Check for even distribution
        if len(df_bonafide) < n_per_class and len(df_spoof) < n_per_class:
            raise ValueError(f'Not enough samples: bonafide={len(df_bonafide)}, spoof={len(df_spoof)}')

        df_bonafide_sampled = df_bonafide.sample(n=n_samples, random_state=42)
        df_spoof_sampled = df_spoof.sample(n=n_samples, random_state=42)
        return pd.concat([df_bonafide_sampled, df_spoof_sampled]).sample(frac=1, random_state=42)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        print(f'===== Getting item {idx} =====')
        row = self.df.iloc[idx]
        file_name = row['file_name']
        label = 1 if row['label'] == 'spoof' else 0
        print(f'file_name: {file_name}, label: {label}')

        audio_path = os.path.join(self.audio_dir, f'{file_name}.flac')
        print(f'audio_path: {audio_path}')
        print(f'Exisits? {os.path.exists(audio_path)}')

        print('1. before sf.read')
        y, sr = sf.read(audio_path)
        print(f'2. after sf.read: y.shape={y.shape}, sr={sr}, dtype={y.dtype}')

        print('3. before tensor conversion')
        waveform = torch.from_numpy(y)
        print(f'4. after tensor conversion: waveform.shape={waveform.shape}')

        # Resampling
        print('5. before resampling check')
        if sr != self.target_sr:
            print(f'5a. resampling of {sr}, to {self.target_sr}')
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(y)
            print(f'5b. after resampling: waveform.shape={waveform.shape}')
        else:
            print(f'5a. resampling not necessary')
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        # Convert to Mono if not
        print(f'6. before mono check')
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            print(f'6a. stereo --> mono, Channels: {waveform.shape[0]}')
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f'6b. after mono check: waveform.shape={waveform.shape}')
        else:
            print(f'6a. already mono')

        # Squeeze 2dim Vector to 1dim ([1,T] --> [T,])
        print(f'7. before squeeze')
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        print(f'8. after squeeze: waveform.shape={waveform.shape}')

        # Apply processor
        inputs = self.processor(y, sampling_rate = self.target_sr, return_tensors='pt')
        input_values = inputs.input_values.squeeze(0)

        return {
            'input_values': input_values,
            'label': torch.tensor(label, dtype=torch.long)
        }