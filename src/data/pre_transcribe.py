"""
Transcribes all ASVspoof 2019 LA audio with Whisper and writes a CSV. ASVDataset
reads this to feed the text branch, so run it once before training and point
TRANSCRIPT_PATH in your .env at the output.
"""

import os
import torch
import pandas as pd
import whisper
import soundfile as sf
import librosa
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DATA_ROOT = os.environ.get("ASV_DATA_ROOT")
assert DATA_ROOT, "Set ASV_DATA_ROOT in your .env file"
SUBSETS = ['train', 'dev']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

whisper_model = whisper.load_model("small", device=DEVICE)
all_transcripts = []

for subset in SUBSETS:
    audio_dir = os.path.join(DATA_ROOT, f'ASVspoof2019_LA_{subset}', 'flac')

    protocol_file = (
        f'ASVspoof2019.LA.cm.{subset}.trn.txt' if subset == 'train'
        else f'ASVspoof2019.LA.cm.{subset}.trl.txt'
    )
    protocol_path = os.path.join(DATA_ROOT, 'ASVspoof2019_LA_cm_protocols', protocol_file)
    df_proto = pd.read_csv(
        protocol_path, sep=' ', header=None,
        names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label']
    )

    print(f"Transcribing {subset} set ({len(df_proto)} files)...")

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
