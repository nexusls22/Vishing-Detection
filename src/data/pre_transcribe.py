import os
import torch
import pandas as pd
import whisper
import soundfile as sf
import librosa
import torch_directml
from tqdm import tqdm


# Paths
DATA_ROOT = r"C:\Users\Luis\Desktop\LA\LA"
SUBSETS = ['train', 'dev']

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch_directml.is_available():
    DEVICE = torch_directml.device()
else:
    DEVICE = torch.device('cpu')

# Load Whisper model
model = whisper.load_model("small")

# Prepare output
all_transcripts = []

for subset in SUBSETS:
    # Path to audio files<
    audio_dir = os.path.join(DATA_ROOT, f'ASVspoof2019_LA_{subset}', 'flac')
    # Path to protocol files, respectively for subsets
    if subset == 'train':
        protocol_path = os.path.join(DATA_ROOT, 'ASVspoof2019_LA_cm_protocols', f'ASVspoof2019.LA.cm.{subset}.trn.txt')
    else:
        protocol_path = os.path.join(DATA_ROOT, 'ASVspoof2019_LA_cm_protocols', f'ASVspoof2019.LA.cm.{subset}.trl.txt')

    df_proto = pd.read_csv(protocol_path, sep=' ', header=None, names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label'])

    file_names = df_proto['file_name'].tolist()

    print(f"Transcribing {subset} set ({len(file_names)} files)...")

    for filename in tqdm(file_names): # Iterables for entries in file_names (Visualization of transcription progress)
        audio_path = os.path.join(audio_dir, f'{filename}.flac')

        try:
            # Load audio
            audio, sr = sf.read(audio_path, dtype='float32')
            if sr != 16000:
                # Resample if needed
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # convert to mono

            # Transcribe
            result = model.transcribe(audio, language='en')
            transcript = result['text'].strip()

        except Exception as e:
            print(f"Error with {filename}: {e}")
            transcript = ""

        all_transcripts.append({
            'file_name': filename,
            'subset': subset,
            'transcript': transcript
        })

# Save to CSV
transcript_df = pd.DataFrame(all_transcripts)
transcript_df.to_csv('../models/training/transcripts.csv', index=False)
print("Transcripts saved to transcripts.csv")