import os
import librosa
import numpy as np
import soundfile as sf
import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor


def augment_audio(y, sr):
    """
    Augment Audio
    Function with various effects to the audio data to keep the pretrained model from shortcutting.
    :param y: 1D array
    :param sr: 1D array
    :return: Augmented audio
    """

    # 1. Random gain (0.1 to 5.0)
    gain = np.random.uniform(0.1, 5.0)
    y = y * gain

    # 2. Additive Gaussian noise (SNR 5 - 25 dB)
    if np.random.rand() < 0.7:
        snr_db = np.random.uniform(5, 25)
        noise_std = np.sqrt(np.var(y)) / (10**(snr_db/20))
        y = y + np.random.normal(0, noise_std, y.shape)

    # 3. Speed perturbation (0.8 - 1.2) - changes duration and pitch
    if np.random.rand() < 0.7:
        speed = np.random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate=speed)

    # 4. Pitch shift (+- 2 semitones)
    if np.random.rand() < 0.5:
        n_steps = np.random.uniform(-2, 2)   # Halbtöne
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # 5. Fixed length cropping/padding (48,000 samples at 16kHz)
    target_len = 48000
    if len(y) > target_len:
        start = np.random.randint(0, len(y) - target_len)
        y = y[start:start+target_len]
    elif len(y) < target_len:
        pad = target_len - len(y)
        pad_left = np.random.randint(0, pad)
        pad_right = pad - pad_left
        y = np.pad(y, (pad_left, pad_right), 'constant')

    return y.astype(np.float32)


class ASVDataset(Dataset):
    """
    Class ASVDataset(Dataset): Dataset for ASVspoof2019 LA dataset
    Custom dataset for ASVspoof data for reading the protocol into a DataFrame, pre-tokenizes, performs stratified sampling and returns a dict of processed data
    Returns:
        Dataset
    """


    def __init__(self, data_root: str, subset: str, processor: Wav2Vec2Processor, text_tokenizer, target_sr: int = 16000, samples: int = None,):
        """
        __init__ method: Initializes the ASVDataset object.
        :param data_root: str = Path to the data folder
        :param subset: str = Subset of the data to use (train, dev, eval)
        :param processor: Processor object from transformers
        :param text_tokenizer: TextTokenizer for later evaluation with transcripts
        :param target_sr: int = KHz sampling rate of the audio
        :param samples: int = Number of samples
        """

        self.data_root = data_root
        self.subset = subset
        self.processor = processor
        self.text_tokenizer = text_tokenizer
        self.target_sr = target_sr

        # Load the protocol file
        protocol_map = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            # 'eval': 'ASVspoof2019.LA.cm.eval.trl.txt',
        }

        protocol_path = os.path.join(self.data_root, 'ASVspoof2019_LA_cm_protocols',
            protocol_map[subset])

        if os.path.exists(protocol_path):
            self.df = pd.read_csv(protocol_path, sep=' ', header=None, names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label'])
        else:
            raise FileNotFoundError(f'Protocol file not found at: {protocol_path}')

        # Attack type labels for dataset
        self.attack_types = sorted(self.df[self.df['label'] == 'spoof']['attack_type'].unique())
        self.attack_to_idx = {at: i for i, at in enumerate(self.attack_types)}
        self.ignore_attack_idx = -1 # Add a special 'ignore' index for bonafide and unkown attacks

        #transcript_path = r'C:\Users\Luis\Griffith\Vishing_Project\Vishing_Detection\src\models\data\training\transcripts.csv'
        transcript_path = r'C:\tmp\vishing_detection\src\models\data\training\transcripts.csv'

        if os.path.exists(transcript_path):
            transcript_df = pd.read_csv(transcript_path)
            self.df = self.df.merge(transcript_df, how='left', on='file_name')
            self.df['transcript'] = self.df['transcript'].fillna('')
        else:
            print(f'Warning: File not found at: {transcript_path}. Transcripts are empty.')
            self.df['transcript'] = ''

        # Optional: Stratified sampling to balance the classes
        if samples:
            self.df = self._stratified_sample(samples)

        # Pre‑tokenize all transcripts once (for efficiency)
        self.transcript_ids = []
        self.transcript_masks = []

        for _, row in self.df.iterrows():
            transcript = row.get('transcript', '')
            encoder = self.text_tokenizer(
                transcript,
                truncation=True,
                max_length=128,
                padding='max_length',  # fixed length for easy stacking
                return_tensors='pt'
                )
            self.transcript_ids.append(encoder['input_ids'].squeeze(0))
            self.transcript_masks.append(encoder['attention_mask'].squeeze(0))

        # Path to the audio files
        self.data_path = os.path.join(self.data_root, f'ASVspoof2019_LA_{subset}', 'flac')
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f'Data path not found at: {self.data_path}')

        # self.transcript_ids = []
        # self.transcript_attention = []

        #for _, row in self.df.iterrows():
            # transcript = row.get('transcript', '')
            # encoder = self.text_tokenizer(transcript, truncation = True, max_length = 128, padding = 'max_length', return_tensors = 'pt')
            # self.transcript_ids.append(encoder['input_ids'].squeeze(0))
            # self.transcript_attention.append(encoder['attention_mask'].squeeze(0))


    def _stratified_sample(self, n_samples: int):
        """
        _stratified_sample: Samples the dataset to ensure an equal distribution of both labels.
        """

        df_bonafide = self.df[self.df['label'] == 'bonafide']
        df_spoof = self.df[self.df['label'] == 'spoof']
        n_per_class = n_samples // 2

        # Check for even distribution
        if len(df_bonafide) < n_per_class or len(df_spoof) < n_per_class:
            raise ValueError(f'Not enough samples: bonafide={len(df_bonafide)}, spoof={len(df_spoof)}')

        df_bonafide_sampled = df_bonafide.sample(n=n_per_class, random_state=42) # n = Number of items form axis to return
        df_spoof_sampled = df_spoof.sample(n=n_per_class, random_state=42)
        balanced_samples = pd.concat([df_bonafide_sampled, df_spoof_sampled]).sample(frac=1, random_state=42) # frac = Fraction of item from axis to return - makes sure to shuffle the rows

        return balanced_samples


    def __len__(self): return len(self.df)


    def __getitem__(self, index):
        """
        __getitem__: Returns an item from the dataset.
        """

        row = self.df.iloc[index] # Indexer - iloc[] integer position (from 0 to length-1)
        file_name = row['file_name']
        # transcript = row.get('transcript', '')
        label = 1 if row['label'] == 'spoof' else 0

        if row['label'] == 'spoof':
            attack = row['attack_type']
            attack_idx = self.attack_to_idx.get(attack, self.ignore_attack_idx)
        else:
            attack_idx = self.ignore_attack_idx # bonfide = ignore

        # Load the audio
        audio_path = os.path.join(self.data_path, f'{file_name}.flac')
        y, sr = sf.read(audio_path)  # y is float64, shape (T,)

        # Resample if needed
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr = sr, target_sr = self.target_sr)
            sr = self.target_sr

        # Convert to mono
        if y.ndim > 1:
            y = y.mean(axis = 1)

        if self.subset == 'train':
            # Apply audio augmentation (only for training, not for dev/eval)
            y = augment_audio(y, sr)

        y = y.astype(np.float32)

        # Process with Wav2Vec2Processor (normalises, extracts features, converts to PyTorch tensors)
        inputs = self.processor(y, sampling_rate=self.target_sr, return_tensors='pt', return_attention_mask = True) # Normalization, Feature extraction (CNN), Conversion to PyTorch ('pt') tensors
        input_values = inputs.input_values.squeeze(0) # # (seq_len,)
        attention_mask = inputs.attention_mask.squeeze(0) # (seq_len,)

        # Return dict of processed data
        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'transcript_ids': self.transcript_ids[index],
            'transcript_mask': self.transcript_masks[index],
            'label': torch.tensor(label, dtype=torch.long), # Dtype torch.long as expected by CrossEntropyLoss - Predicted probabilities compared to true labels
            'attack_idx': torch.tensor(attack_idx, dtype=torch.long)
        }