"""
PyTorch Dataset for the ASVspoof 2019 LA corpus. Reads the protocol file, merges
the pre-computed Whisper transcripts, pre-tokenises them at init, and returns raw
waveforms plus tokenised transcripts and labels. Padding, augmentation and
normalisation happen later in collate_fn.
"""

from __future__ import annotations

import os
import librosa
import numpy as np
import soundfile as sf
import torch
import pandas as pd
import torchaudio.functional as taF
import torch.nn.functional as nnF

from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor  # only used for type hints, not at runtime

load_dotenv()


def augment_audio(y, sr):
    """
    Randomised augmentation chain for a 16 kHz waveform: gain (always), additive
    noise (p=0.7), speed perturbation (p=0.7), pitch shift (p=0.5), then crop/pad
    to 48000 samples. Returns a 1-D float32 array.
    """

    y = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.float()

    gain = np.random.uniform(0.1, 5.0)
    y = y * gain

    if np.random.rand() < 0.7:
        snr_db = np.random.uniform(5, 25)
        noise_std = torch.std(y) / (10 ** (snr_db / 20))
        y = y + torch.randn_like(y) * noise_std

    if np.random.rand() < 0.7:
        speed = np.random.uniform(0.8, 1.2)
        y, _ = taF.speed(y.unsqueeze(0), sr, speed)
        y = y.squeeze(0)

    if np.random.rand() < 0.5:
        n_steps = np.random.randint(-2, 3)
        y = taF.pitch_shift(y.unsqueeze(0), sr, n_steps).squeeze(0)

    target_len = 48_000
    if y.size(0) > target_len:
        start = np.random.randint(0, y.size(0) - target_len)
        y = y[start:start + target_len]
    elif y.size(0) < target_len:
        pad = target_len - y.size(0)
        pad_left = np.random.randint(0, pad)
        pad_right = pad - pad_left
        y = nnF.pad(y, (pad_left, pad_right))

    return y.cpu().numpy().astype(np.float32)


class ASVDataset(Dataset):
    """
    ASVspoof 2019 LA train/dev dataset. __getitem__ returns a dict with the raw
    waveform, the pre-tokenised transcript ids and mask, the binary label
    (0 bonafide, 1 spoof) and the attack-type index (-1 for bonafide).
    """

    def __init__(self, data_root: str, subset: str, processor: Wav2Vec2Processor,
                 text_tokenizer, target_sr: int = 16000, samples: int = None):
        self.data_root = data_root
        self.subset = subset
        self.processor = processor
        self.text_tokenizer = text_tokenizer
        self.target_sr = target_sr

        protocol_map = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
        }
        protocol_path = os.path.join(
            self.data_root, 'ASVspoof2019_LA_cm_protocols', protocol_map[subset]
        )
        if not os.path.exists(protocol_path):
            raise FileNotFoundError(f'Protocol file not found: {protocol_path}')

        self.df = pd.read_csv(
            protocol_path, sep=' ', header=None,
            names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label']
        )

        # Sorted so the attack-type indices are stable across runs.
        self.attack_types = sorted(self.df[self.df['label'] == 'spoof']['attack_type'].unique())
        self.attack_to_idx = {at: i for i, at in enumerate(self.attack_types)}
        self.ignore_attack_idx = -1  # ignore bonafide samples, no attack type included

        transcript_path = os.environ.get('TRANSCRIPT_PATH')
        if not transcript_path:
            raise ValueError('Set TRANSCRIPT_PATH in your .env file')

        if os.path.exists(transcript_path):
            transcript_df = pd.read_csv(transcript_path)
            self.df = self.df.merge(transcript_df, how='left', on='file_name')
            self.df['transcript'] = self.df['transcript'].fillna('')
        else:
            print(f'Warning: transcript file not found at {transcript_path}. Using empty strings.')
            self.df['transcript'] = ''

        if samples:
            self.df = self._stratified_sample(samples)

        # Tokenise once up front rather than per batch in the training loop.
        self.transcript_ids   = []
        self.transcript_masks = []
        for _, row in self.df.iterrows():
            enc = self.text_tokenizer(
                row.get('transcript', ''),
                truncation=True, max_length=128,
                padding='max_length', return_tensors='pt'
            )
            self.transcript_ids.append(enc['input_ids'].squeeze(0))
            self.transcript_masks.append(enc['attention_mask'].squeeze(0))

        self.data_path = os.path.join(self.data_root, f'ASVspoof2019_LA_{subset}', 'flac')
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f'Audio directory not found: {self.data_path}')

    def _stratified_sample(self, n_samples: int) -> pd.DataFrame:
        """Returns a balanced subset with n_samples // 2 examples per class."""
        df_bonafide = self.df[self.df['label'] == 'bonafide']
        df_spoof = self.df[self.df['label'] == 'spoof']
        n_per_class = n_samples // 2

        if len(df_bonafide) < n_per_class or len(df_spoof) < n_per_class:
            raise ValueError(
                f'Not enough samples: bonafide={len(df_bonafide)}, spoof={len(df_spoof)}'
            )

        return pd.concat([
            df_bonafide.sample(n=n_per_class, random_state=42),
            df_spoof.sample(n=n_per_class, random_state=42),
        ]).sample(frac=1, random_state=42)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = 1 if row['label'] == 'spoof' else 0
        attack_idx = (
            self.attack_to_idx.get(row['attack_type'], self.ignore_attack_idx)
            if row['label'] == 'spoof' else self.ignore_attack_idx
        )

        audio_path = os.path.join(self.data_path, f"{row['file_name']}.flac")
        y, sr = sf.read(audio_path)

        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        if y.ndim > 1:
            y = y.mean(axis=1)

        return {
            'waveform': torch.from_numpy(y.astype(np.float32)),
            'transcript_ids': self.transcript_ids[index],
            'transcript_mask': self.transcript_masks[index],
            'label': torch.tensor(label, dtype=torch.long),
            'attack_idx': torch.tensor(attack_idx, dtype=torch.long),
        }
