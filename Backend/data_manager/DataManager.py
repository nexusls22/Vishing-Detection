from pathlib import Path

import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Literal


class DataManager:

    def __init__(self, data_folder_path: str, subset: Literal['train', 'dev', 'eval'], input_sample_size: int):

        self.data_folder_path = Path(data_folder_path)
        self.subset = subset
        self.input_sample_size = input_sample_size

        sub_folder = f'ASVspoof2019_LA_{subset}'
        self.audio_path = self.data_folder_path / sub_folder / 'flac'

        protool_mapping = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }

        protocol_file = protool_mapping[subset]
        protocol_path = self.data_folder_path / 'ASVspoof2019_LA_cm_protocols' / protocol_file
        self.train_protocol_path = protocol_path


    # Helper
    def load_protocol(self) -> pd.DataFrame:
        return pd.read_csv(
            self.train_protocol_path,
            sep=' ',
            header=None,
            names=['speaker_id', 'file_name', 'system_id', 'attack_type', 'label']
        )

    # Check directory
    def check_directory(self, path: str, description: str = "Directory") -> None:
        print(f"\n{description} ({path}):")
        print("-" * 40)
        try:
            for item in sorted(os.listdir(path)):
                print(item)
        except FileNotFoundError:
            print(f"Directory not found: {path}")
        print("-" * 40)

    # Extract features
    def extract_audio_features(self, y: np.ndarray, sr: int, filename: str) -> dict:
        features = {'filename': filename}

        # MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])

        # Spectral features
        features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        features['contrast_mean'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)[0])
        features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])

        # Energy (RMS) – mean over all frames
        features['rms_mean'] = np.mean(librosa.feature.rms(y=y)[0])

        # Pitch
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitches_flat = pitches[pitches > 0]
        features['pitch_mean'] = np.mean(pitches_flat) if len(pitches_flat) > 0 else 0.0

        return features

    def extract_feature_array(self, df):
        # f64 features
        f64 = np.array([
            df['duration'].values,
            df['spectral_centroid_mean'].values,
            df['spectral_bandwidth_mean'].values,
            df['rolloff_mean'].values,
            df['zero_crossing_rate_mean'].values
        ], dtype=np.float64)

        # f32 features
        f32 = np.array([
            df['mean_amplitude'].values,
            df['std_amplitude'].values,
            df['max_amplitude'].values,
            df['min_amplitude'].values,
            df['mfcc_3_mean'].values,
            df['mfcc_3_std'].values,
            df['mfcc_6_mean'].values,
            df['mfcc_6_std'].values,
            df['mfcc_9_mean'].values,
            df['mfcc_9_std'].values,
            df['mfcc_12_mean'].values,
            df['mfcc_12_std'].values
        ], dtype=np.float32)

        # Transposition from (samples, features) to (features, samples)
        return f64.T, f32.T

    #Load data
    def load_data(self) -> pd.DataFrame:

        protocol_df = self.load_protocol()

        # Separate file lists by label
        bonafide_files = protocol_df[protocol_df['label'] == 'bonafide']['file_name'].tolist()
        spoof_files = protocol_df[protocol_df['label'] == 'spoof']['file_name'].tolist()

        # Balanced sampling
        n_per_class = self.input_sample_size // 2
        if len(bonafide_files) < n_per_class or len(spoof_files) < n_per_class:
            print("Warning: One class has fewer files than requested. Adjusting sample.")
            n_per_class = min(len(bonafide_files), len(spoof_files), n_per_class)

        selected_bonafide = random.sample(bonafide_files, n_per_class)
        selected_spoof = random.sample(spoof_files, n_per_class)
        selected_files = selected_bonafide + selected_spoof
        random.shuffle(selected_files)

        basic_data = []
        all_features = []

        print(f"Loading {len(selected_files)} files...")

        for idx, fname in enumerate(selected_files):
            try:
                file_path = os.path.join(self.audio_path, fname + '.flac')
                y, sr = sf.read(file_path, dtype='float32')

                # Convert to mono
                if y.ndim > 1:
                    y = y.mean(axis=1)

                # Resample to 16 kHz if needed
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000

                # Get metadata from protocol
                row = protocol_df[protocol_df['file_name'] == fname].iloc[0]
                label = row['label']
                attack = row['attack_type'] if label == 'spoof' else None  # bonafide → attack = None

                basic_data.append({
                    'filename': fname,
                    'label': label,
                    'attack': attack,
                    'duration': len(y) / sr,
                    'mean_amplitude': np.mean(y),
                    'std_amplitude': np.std(y),
                    'max_amplitude': np.max(y),
                    'min_amplitude': np.min(y)
                })

                feat_dict = self.extract_audio_features(y, sr, fname)
                all_features.append(feat_dict)

            except Exception as e:
                print(f"Error with {fname}: {e}")

        if not basic_data:
            print("No valid audio files found.")
            return pd.DataFrame()

        basic_df = pd.DataFrame(basic_data)
        features_df = pd.DataFrame(all_features)
        full_df = basic_df.merge(features_df, on='filename')

        print(f"\nLoaded: {len(full_df)} rows, {len(full_df.columns)} columns.")
        return full_df

    #Check data
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        report = {}

        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)

        # 1. Missing values per column
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            print("\nColumns with missing values:")
            for col, count in missing.items():
                print(f"   {col}: {count} missing ({count/len(df):.1%})")
        else:
            print("\nNo missing values.")
        report['missing_values'] = missing.to_dict()

        # 2. Special attack column check
        bonafide = df[df['label'] == 'bonafide']
        spoof = df[df['label'] == 'spoof']

        attack_missing_in_bonafide = bonafide['attack'].isnull().all()
        attack_not_missing_in_spoof = spoof['attack'].notnull().all()

        print("\nAttack column check:")
        if attack_missing_in_bonafide:
            print("   bonafide: all attack = NaN (correct)")
        else:
            n_non_nan = bonafide['attack'].notnull().sum()
            print(f"   bonafide: {n_non_nan} entries have an attack value (should be NaN)")

        if attack_not_missing_in_spoof:
            print("   spoof: all attack are filled (correct)")
        else:
            n_nan = spoof['attack'].isnull().sum()
            print(f"   spoof: {n_nan} entries have NaN (should have an attack type)")

        report['attack_check'] = {
            'bonafide_attack_nan': attack_missing_in_bonafide,
            'spoof_attack_not_nan': attack_not_missing_in_spoof
        }

        # 3. Duplicates (by filename)
        duplicate_files = df.duplicated(subset=['filename'], keep=False)
        n_duplicates = duplicate_files.sum()
        if n_duplicates > 0:
            print(f"\n{n_duplicates} duplicate filenames found.")
            report['duplicate_filenames'] = n_duplicates
        else:
            print("\nNo duplicate filenames.")

        # 4. Outlier detection (IQR) for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}

        print("\nOutliers (IQR method) in numeric columns:")
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                outlier_summary[col] = len(outliers)
                print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df):.1%})")
        report['outliers_iqr'] = outlier_summary

        # 5. Data types check
        print("\nData types (excerpt):")
        print(df.dtypes)
        report['dtypes'] = df.dtypes.to_dict()

        print("\n" + "="*60)
        return report


    # ----------------------------------------------------------------------
    # EDA methods (KI)
    # ----------------------------------------------------------------------
    def summary_statistics(self, df: pd.DataFrame, group_by_label: bool = True) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print("\nOverall descriptive statistics:")
        print(df[numeric_cols].describe())

        if group_by_label and 'label' in df.columns:
            print("\nGrouped by label:")
            for label in df['label'].unique():
                subset = df[df['label'] == label]
                print(f"\n--- {label.upper()} ---")
                print(subset[numeric_cols].describe())

    def plot_feature_distributions(self, df: pd.DataFrame, features: List[str], 
                                    bins: int = 50, figsize: Tuple[int, int] = (15, 10)) -> None:
        if 'label' not in df.columns:
            print("Column 'label' missing.")
            return

        n_features = len(features)
        fig, axes = plt.subplots(2, n_features, figsize=figsize)

        for i, feat in enumerate(features):
            if feat not in df.columns:
                continue

            # Histogram
            ax = axes[0, i]
            for label, color in zip(['bonafide', 'spoof'], ['green', 'red']):
                subset = df[df['label'] == label][feat].dropna()
                ax.hist(subset, bins=bins, alpha=0.5, label=label, color=color)
            ax.set_title(feat)
            ax.legend()

            # Boxplot
            ax = axes[1, i]
            data = [df[df['label'] == label][feat].dropna() for label in ['bonafide', 'spoof']]
            ax.boxplot(data, labels=['bonafide', 'spoof'])
            ax.set_title(feat)

        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self, df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True)
        plt.title('Correlation matrix of numeric features')
        plt.show()

    def plot_attack_distribution(self, df: pd.DataFrame) -> None:
        spoof_df = df[df['label'] == 'spoof']
        if spoof_df.empty:
            print("No spoof data available.")
            return

        attack_counts = spoof_df['attack'].value_counts()
        plt.figure(figsize=(10, 6))
        attack_counts.plot(kind='bar')
        plt.title('Attack type distribution (spoof)')
        plt.xlabel('Attack type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()