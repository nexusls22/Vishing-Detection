"""
pre_transcribe.py
One-time script that transcribes all ASVspoof 2019 LA audio files with Whisper
and saves the results to a CSV.

The transcript CSV is required by ASVDataset so that the text branch of the
multimodal model has input. Run this once before training; the output path
should be set as TRANSCRIPT_PATH in your .env file.
"""

import os
import torch
import pandas as pd
import whisper
import soundfile as sf
import librosa
from tqdm import tqdm

DATA_ROOT = r"C:\Users\Luis\Desktop\LA\LA"
SUBSETS   = ['train', 'dev']
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

whisper_model   = whisper.load_model("small", device=DEVICE)
all_transcripts = []

for subset in SUBSETS:
    audio_dir = os.path.join(DATA_ROOT, f'ASVspoof2019_LA_{subset}', 'flac')

    protocol_file = (
        f'ASVspoof2019.LA.cm.{subset}.trn.txt' if subset == 'train'
        else f'ASVspoof2019.LA.cm.{subset}.trl.txt'
    )
    protocol_path = os.path.join(DATA_ROOT, 'ASVspoof2019_LA_cm_protocols', protocol_file)
    df_proto      = pd.read_csv(
        protocol_path, sep=' ', header=None,
        names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label']
    )

    print(f"Transcribing {subset} set ({len(df_proto)} files)…")

    for filename in tqdm(df_proto['file_name'].tolist()):
        audio_path = os.path.join(audio_dir, f'{filename}.flac')
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            result     = whisper_model.transcribe(audio, language='en')
            transcript = result['text'].strip()
        except Exception as e:
            print(f"Error with {filename}: {e}")
            transcript = ''

        all_transcripts.append({'file_name': filename, 'subset': subset, 'transcript': transcript})

transcript_df = pd.DataFrame(all_transcripts)
transcript_df.to_csv('../models/training/transcripts.csv', index=False)
print("Transcripts saved to transcripts.csv")
