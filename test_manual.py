# test_manual.py
from src.data.asvdataset import ASVDataset
from transformers import Wav2Vec2Processor
from src.data.collate import collate_fn

data_root = '/Users/luissander/Griffith-Local/SecondSemester/vishing-detection/data'
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
dataset = ASVDataset(data_root, 'train', processor, max_samples=2)

item0 = dataset[0]
item1 = dataset[1]
print("Items loaded")

item0_copy = {
    'input_values': item0['input_values'].clone(),
    'label': item0['label'].clone()
}
item1_copy = {
    'input_values': item1['input_values'].clone(),
    'label': item1['label'].clone()
}
print("Items kopiert")

batch = collate_fn([item0_copy, item1_copy])
print('Batch created, shape:', batch['input_values'].shape)