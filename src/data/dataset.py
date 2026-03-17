# Dataset class (path where data is found, subset for right protocol to dataset, processor for computing, sample rate has to 16KHz, number of max samples)
import torch
import torchaudio

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

class ASVDataset(Dataset):
    def __init__(self, data_root: str, subset: str, processor: Wav2Vec2Processor, target_sr: int = 16000, max_samples: int = None):
        self.data_root = data_root
        self.subset = subset
        self.processor = processor
        self.target_sr = target_sr

    # Map path to right protocol for sample set
    protocol_path = self.data_root / 'protocols/ASVspoof2019_LA_cm_protocols' / {
        'train': 'ASVspoof2019.LA.cm.train.trn.txt',
        'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
        'eval': 'ASVspoof2019.LA.cm.eval.trl.txt',
    }[subset]

    self.df = pd.read_csv(protocol_path, sep=' ', header=None, names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label'])

    if max_samples:
        self.df = self._stratified_sample(max_samples)

    self.audio_dir= self.data_root / 'raw' / f'ASVspoof2019_LA_{subset}' / 'flac'

    # Stratifying samples for even distribution of both labels
    def _stratified_sample(self, n_samples: int):
        df_bonafide = self.df[self.df['label'] == 'bonafide']
        df_spoof = self.df[self.df['label'] == 'spoof']
        n_per_class = n_per_class // 2

        # Check for even distribution
        if len(df_bonafide) < n_samples and len(df_spoof) < n_samples:
            raise ValueError(f'Not enough samples: bonafide={len(df_bonafide)}, spoof={len(df_spoof)}')

        df_bonafide_sampled = df_bonafide.sample(n=n_samples, random_state=42)
        df_spoof_sampled = df_spoof.sample(n=n_samples, random_state=42)
        return pd.concat([df_bonafide_sampled, df_spoof_sampled].sample(frac=1, random_state=42))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']
        label = 1 if row['label'] == 'spoof' else 0

        audio_path = self.audio_dir / f'{file_name}.flac'
        waveform, sr = torchaudio.load(audio_path)

        # Resampling
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Convert to Mono if not
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Squeeze 2dim Vector to 1dim ([1,T] --> [T,])
        waveform = waveform.unsqueeze(0)

        # Apply processor
        inputs = self.processor(waveform, sampling_rate = self.target_sr, return_tensors='pt')
        input_values = inputs.input_values.squeeze(0)

        return {
            'input_values': input_values,
            'label': torch.tensor(label, dtype=torch.long)
        }