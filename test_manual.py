# test_manual.py
from src.data.asvdataset import ASVDataset
from transformers import Wav2Vec2Processor
from src.data.collate import collate_fn

data_root = 'C:/Users/Luis/Desktop/LA/LA'
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
dataset = ASVDataset(data_root, 'train', processor, max_samples=2)

item0 = dataset[0]
item1 = dataset[1]
print("Items loaded")

batch = collate_fn([item0, item1])
print('Batch created, shape:', batch['input_values'].shape)