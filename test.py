import os
from transformers import Wav2Vec2Processor
from src.data.ASVDataset import ASVDataset

data_root = 'C:/Users/Luis/Desktop/LA/LA'
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

# Create train and dev datasets
train_ds = ASVDataset(data_root, 'train', processor)
dev_ds = ASVDataset(data_root, 'dev', processor)

print("\n" + "="*60)
print("TRAIN SET")
print("="*60)
print("Label counts:\n", train_ds.df['label'].value_counts())
print("First 5 rows:\n", train_ds.df.head())
print("Sample first 10 labels from __getitem__:")
for i in range(10):
    sample = train_ds[i]
    print(f"  Index {i}: label = {sample['label'].item()}, file = {train_ds.df.iloc[i]['file_name']}")

print("\n" + "="*60)
print("DEV SET")
print("="*60)
print("Label counts:\n", dev_ds.df['label'].value_counts())
print("First 5 rows:\n", dev_ds.df.head())
print("Sample first 10 labels from __getitem__:")
for i in range(10):
    sample = dev_ds[i]
    print(f"  Index {i}: label = {sample['label'].item()}, file = {dev_ds.df.iloc[i]['file_name']}")