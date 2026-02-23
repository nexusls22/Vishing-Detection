import numpy as np
import pandas as pd
import soundfile as sf
import os
import librosa
from pathlib import Path
from itables import init_notebook_mode, show
from itables.widget import ITable
from pandas import DataFrame
from pandas.io.parsers import TextFileReader


class DataManager:

    init_notebook_mode(connected=True)
    data_folder_path = ''

    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.train_data_path = self.data_folder_path + 'ASVspoof2019_LA_train/flac'
        self.train_protocol_path = self.data_folder_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

    def check_data(self):
        elements_in_dir = []

        print('Data Directory Contents:')
        print('------------------------')

        for i in os.listdir(self.data_folder_path):
            elements_in_dir.append(i)
            print(i)

        print('------------------------')





    def load_protocol(self):
        return pd.read_csv(self.train_protocol_path, sep=' ', header = None, names = ['speaker_id', 'file_name', 'system_id', 'attack_type','key'])

    def load_data(self):
        test_df = self.load_protocol()

        results = []

        test_bonafide_df = test_df[test_df['key'] == 'bonafide']['file_name'].iloc[:len(test_df)].tolist()
        test_spoof_df = test_df[test_df['key'] == 'spoof']['file_name'].iloc[:len(test_df)].tolist()
        test_samples = test_bonafide_df + test_spoof_df

        for f in test_samples[:10]: #Input method to set number of samples
            try:
                file_path = os.path.join(self.train_data_path, f + '.flac')
                y, sr = sf.read(file_path, dtype='float32')

                if y.ndim > 1:
                    y = y.mean(axis=1)

                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000

                key = test_df[test_df['file_name'] == f]['key'].values[0]
                attack = test_df[test_df['file_name'] == f]['attack_type'].values[0]

                results.append({
                    'filename': f,
                    'label': key,
                    'attack': attack,
                    'duration': len(y) / sr,
                    'mean': y.mean(),
                    'std': y.std(),
                    'max': y.max(),
                    'min': y.min()
                })

            except Exception as e:
                print(f"{f}: {e}")

        if results:
            results_df = pd.DataFrame(results)
            return results_df
        else:
            return None

    def preprocess_data(self):
        """Preprocess data for training and evaluation."""

