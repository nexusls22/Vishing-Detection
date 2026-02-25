import pandas as pd
import soundfile as sf
import numpy as np
import os
import librosa
from itables import init_notebook_mode


class DataManager:

    init_notebook_mode(connected=True)
    data_folder_path = ''
    results = []
    input_sample_size = int(input("Please enter the number of samples to be tested: "))


    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.train_data_path = self.data_folder_path + 'ASVspoof2019_LA_dev/flac'
        self.train_protocol_path = self.data_folder_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

    def load_protocol(self):
        return pd.read_csv(self.train_protocol_path, sep=' ', header = None, names = ['speaker_id', 'file_name', 'system_id', 'attack_type','key'])

    def check_data_dir(self):
        elements_in_dir = []

        print('Data Directory Contents:')
        print('------------------------')

        for i in os.listdir(self.data_folder_path):
            elements_in_dir.append(i)
            print(i)

        print('------------------------')

    def check_data_files(self):
        elements_in_dir = []

        print('Data Files:')
        print('------------------------')

        for i in os.listdir(self.train_data_path):
            elements_in_dir.append(i)
            print(i)

        print('------------------------')

    def load_data(self, results, input_sample_size):
        test_df = self.load_protocol()
        features_dict = {}

        #Sample lists
        test_bonafide_df = test_df[test_df['key'] == 'bonafide']['file_name'].iloc[:len(test_df)].tolist()
        test_spoof_df = test_df[test_df['key'] == 'spoof']['file_name'].iloc[:len(test_df)].tolist()
        test_samples = test_bonafide_df + test_spoof_df

        #Check for valid audio files
        if not test_bonafide_df:
            print("No bonafide samples found.")
        if not test_spoof_df:
            print("No spoof samples found.")

        for f in test_samples[:input_sample_size]:
            try:
                file_path = os.path.join(self.train_data_path, f + '.flac')
                y, sr = sf.read(file_path, dtype='float32')

                #Convert to mono sound
                if y.ndim > 1:
                    y = y.mean(axis=1)

                #Resample to 16kHz
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000

                key = test_df[test_df['file_name'] == f]['key'].values[0]
                attack = test_df[test_df['file_name'] == f]['attack_type'].values[0]

                #Extract features and append to results
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
                features_dict = self.extract_features(y, sr)

            except Exception as e:
                print(f"{f}: {e}")



        #Build dataframe out of results
        if results and features_dict:
            results_df = pd.DataFrame(results)
            features_df = pd.DataFrame.from_dict(features_dict)

            return results_df, features_df
        else:
            return print("No valid audio files found.")

    def extract_features(self, y, sr):

        features = {}

        #MFCC (Mel Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc__{i}__mean'] = np.mean(mfcc[i])
            features[f'mfcc__{i}__std'] = np.std(mfcc[i])

        #Spectral features
        features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        features['contrast_mean'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)[0])
        features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])

        #Energy and RMS (root-mean-square)
        features['rms'] = librosa.feature.rms(y=y)[0]

        #Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        features['pitches_mean'] = np.mean(pitches[pitches > 0] if np.any(pitches > 0) else 0)



        return features


    def check_data_quality(self, features_df):

        # Search for missing entries
        missing_entries = features_df.isnull().sum()
        if not missing_entries.empty:
            print(f"Found {len(missing_entries)} missing entries:")
            print(missing_entries)
        else:
            print("No missing entries found.")

        #Search for duplicate entrie
        duplicate_entries = features_df.duplicated(subset=['filename'], keep=False)
        if not duplicate_entries.empty:
            print(f"Found {len(duplicate_entries)} duplicate entries:")
            print(duplicate_entries)
        else:
            print("No duplicate entries found.")

        #Search for outliers

    def preprocess_data(self):
        """Preprocess data fortraining and evaluation."""
        return

