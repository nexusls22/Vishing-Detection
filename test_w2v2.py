from torch.utils.data import DataLoader
from src.data.asvdataset import ASVDataset
from src.data.collate import collate_fn
from transformers import Wav2Vec2Processor

data_root = '/Users/luissander/Griffith-Local/SecondSemester/vishing-detection/data'
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

dataset = ASVDataset(data_root, 'train', processor, max_samples=10)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)

for i, batch in enumerate(dataloader):
    print(f"Batch {i}: {batch['input_values'].shape}")
    break