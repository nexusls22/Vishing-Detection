"""
build_prototypes.py
One-time script that computes class prototype vectors from the training set
and saves them to src/models/prototypes.pt for use by the Gradio app.

The prototype approach reconstructs the AAM-Softmax scoring geometry:
each class is represented by the normalised mean of its training embeddings.
At inference time, a recording is classified by comparing its embedding to
these two prototypes via cosine similarity.

The EER threshold (from reconstruct_eer.py) is embedded in the output file
so the app loads it automatically.

Run from the project root before launching the app:
    python build_prototypes.py
"""

import os
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoTokenizer

from src.data.ASVDataset import ASVDataset
from src.data.collate import collate_fn
from src.models.models import MultimodalVishingDetector

load_dotenv()

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT  = os.environ.get("ASV_DATA_ROOT")
assert DATA_ROOT, "Set ASV_DATA_ROOT in your .env file"

BATCH_SIZE = 64
THRESHOLD  = 0.6297   # cosine EER threshold obtained from reconstruct_eer.py
OUT_PATH   = "src/models/prototypes.pt"

print(f"Device : {DEVICE}")

model = MultimodalVishingDetector(embed_dim=256).to(DEVICE)
state = torch.load("src/models/best_model.pth", map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

processor = Wav2Vec2Processor.from_pretrained(
    'facebook/wav2vec2-base', padding='max_length', max_length=16000 * 4, truncation=True
)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Full training set (unbalanced) matches the approach used in reconstruct_eer.py
train_ds = ASVDataset(DATA_ROOT, 'train', processor, tokenizer)
loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn, num_workers=0, pin_memory=False)
print(f"Samples: {len(train_ds)}")

emb_list, label_list = [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Building prototypes"):
        iv = batch['input_values'].to(DEVICE)
        am = batch['attention_mask'].to(DEVICE)
        ti = batch['transcript_ids'].to(DEVICE)
        tm = batch['transcript_mask'].to(DEVICE)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            _, _, emb, _, _ = model(iv, am, ti, tm, return_embeddings=True)

        emb_list.append(emb.float().cpu())
        label_list.append(batch['labels'].cpu())

embeddings = torch.cat(emb_list,   dim=0)  # (N, 256)
labels     = torch.cat(label_list, dim=0)  # (N,)

# Normalised mean embedding per class — approximates the AAM-Softmax weight vectors
emb_norm       = F.normalize(embeddings, dim=1)
proto_bonafide = F.normalize(emb_norm[labels == 0].mean(dim=0, keepdim=True), dim=1)
proto_spoof    = F.normalize(emb_norm[labels == 1].mean(dim=0, keepdim=True), dim=1)

sep = F.cosine_similarity(proto_bonafide, proto_spoof).item()
print(f"\nPrototype cosine similarity : {sep:.4f}  (lower = better separated)")
print(f"Bonafide : {(labels == 0).sum().item()} samples")
print(f"Spoof    : {(labels == 1).sum().item()} samples")

torch.save({'bonafide': proto_bonafide, 'spoof': proto_spoof, 'threshold': THRESHOLD}, OUT_PATH)
print(f"\nPrototypes saved to {OUT_PATH}")
print(f"Threshold : {THRESHOLD} (from reconstruct_eer.py dev-set EER)")
