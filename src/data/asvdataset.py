# Dataset class (path where data is found, subset for right dataset protocol, processor for computing,
#               sample rate has to be 16KHz, number of max samples)
import os

import librosa
import numpy as np
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
        protocol_path = os.path.join(self.data_root, 'ASVspoof2019_LA_cm_protocols', {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt',
        }[subset])

        self.df = pd.read_csv(protocol_path, sep=' ', header=None, names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label'])
        print(f"Label distribution for {self.subset}:")
        print(self.df['label'].value_counts())
        print(self.df.head(10))
        print(self.df['label'].unique())

        transcript_path = r'C:\Users\Luis\Griffith\Vishing_Project\Vishing_Detection\models\data\training\transcripts.csv'
        if os.path.exists(transcript_path):
            transcript_df = pd.read_csv(transcript_path)
            self.df = self.df.merge(transcript_df, how='left', on='file_name')
        else:
            print(f'Warning: File not found at: {transcript_path}.  Transcripts are empty.')
            self.df['transcript'] = ''

        if max_samples:
            self.df = self._stratified_sample(max_samples)

        self.audio_dir = os.path.join(self.data_root, f'ASVspoof2019_LA_{subset}', 'flac')
        print(self.audio_dir)

        print(f'DEBUG: audio path: {self.audio_dir}')


    # Stratifying samples for even distribution of both labels
    def _stratified_sample(self, n_samples: int):
        df_bonafide = self.df[self.df['label'] == 'bonafide']
        df_spoof = self.df[self.df['label'] == 'spoof']
        n_per_class = n_samples // 2

        # Check for even distribution
        if len(df_bonafide) < n_per_class and len(df_spoof) < n_per_class:
            raise ValueError(f'Not enough samples: bonafide={len(df_bonafide)}, spoof={len(df_spoof)}')

        df_bonafide_sampled = df_bonafide.sample(n=n_per_class, random_state=42)
        df_spoof_sampled = df_spoof.sample(n=n_per_class, random_state=42)
        return pd.concat([df_bonafide_sampled, df_spoof_sampled]).sample(frac=1, random_state=42)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']
        transcript = row.get('transcript', '')
        label = 1 if row['label'] == 'spoof' else 0

        audio_path = os.path.join(self.audio_dir, f'{file_name}.flac')
        y, sr = sf.read(audio_path)  # y is float64, shape (T,)

        # Resample if needed
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # Ensure mono (should be already)
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Convert to float32
        y = y.astype(np.float32)

        # For processor: we need the audio array
        inputs = self.processor(y, sampling_rate=self.target_sr, return_tensors='pt')
        input_values = inputs.input_values.squeeze(0)

        # For Whisper: we also need the raw audio (already resampled, float32)
        raw_audio_for_whisper = y.copy()

        return {
            'input_values': input_values,
            'transcript': transcript,
            'raw_audio_for_whisper': raw_audio_for_whisper,
            'label': torch.tensor(label, dtype=torch.long)
        }