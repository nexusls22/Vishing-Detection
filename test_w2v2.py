import os.path
import platform
import torch

from transformers import Wav2Vec2Processor
from src.data.asvdataset import ASVDataset
from src.data.collate import collate_fn
from torch.utils.data import DataLoader

data_root = None

if data_root is None:
    system = platform.system()
    data_root = (
        os.path.join(os.path.abspath(os.sep), 'Users', 'Luis', 'Desktop','LA', 'LA')
        if system == 'Windows'
        else os.path.join('data')
    )
    print(f'Using {system} OS')

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')

dataset = ASVDataset(data_root, 'train', processor, max_samples=10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)