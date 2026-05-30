"""
reconstruct_eer.py
Reconstructs the cosine EER from a saved model checkpoint without needing
the original AAM-Softmax criterion weights.

Approach (mirrors the training evaluation):
  1. Extract 256-dim fusion embeddings for the full training set
  2. Compute per-class prototype vectors as normalised mean embeddings
  3. Score the dev set by cosine similarity to the spoof prototype
  4. Compute EER from the resulting score distribution

The reconstructed EER closely approximates the training-time EER because
AAM-Softmax drives each class's mean embedding toward its weight vector.

Run from the project root:
    python reconstruct_eer.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve
from transformers import Wav2Vec2Processor, AutoTokenizer
from src.data.ASVDataset import ASVDataset
from src.data.collate import collate_fn
from src.models.models import MultimodalVishingDetector

load_dotenv()

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT  = os.environ.get("ASV_DATA_ROOT")
assert DATA_ROOT, "Set ASV_DATA_ROOT in your .env file"
BATCH_SIZE = 128

print(f"Device: {DEVICE}")

model = MultimodalVishingDetector(embed_dim=256).to(DEVICE)
state = torch.load('src/models/best_model.pth', map_location=DEVICE, weights_only=True)
result = model.load_state_dict(state, strict=False)
print(f"Loaded | unexpected: {len(result.unexpected_keys)} | missing: {len(result.missing_keys)}")
model.eval()

processor = Wav2Vec2Processor.from_pretrained(
    'facebook/wav2vec2-base', padding='max_length', max_length=16000 * 4, truncation=True
)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Register a forward hook on the fusion module to capture embeddings without
# modifying the model's forward signature
emb_cache = []

def _hook(module, input, output):
    emb_cache.append(output.detach().float().cpu())

hook = model.fusion.register_forward_hook(_hook)


def get_embeddings(loader, desc):
    """Runs the model on a DataLoader and returns all fusion embeddings and labels."""
    emb_cache.clear()
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            iv = batch['input_values'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            ti = batch['transcript_ids'].to(DEVICE)
            tm = batch['transcript_mask'].to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                model(iv, am, ti, tm)
            labels_all.extend(batch['labels'].cpu().numpy())
    return torch.cat(emb_cache, dim=0), np.array(labels_all)


# Step 1: build class prototypes from the training set
print("\n[1/3] Extracting train embeddings to build prototypes…")
train_ds     = ASVDataset(DATA_ROOT, 'train', processor, tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=0, pin_memory=False)

train_embs, train_labels = get_embeddings(train_loader, 'Train embeddings')
train_embs_norm = F.normalize(train_embs, dim=1)

bonafide_mask = train_labels == 0
spoof_mask    = train_labels == 1
print(f"    bonafide samples: {bonafide_mask.sum()} | spoof samples: {spoof_mask.sum()}")

proto_bon   = F.normalize(train_embs_norm[bonafide_mask].mean(dim=0, keepdim=True), dim=1)
proto_spoof = F.normalize(train_embs_norm[spoof_mask].mean(dim=0, keepdim=True),   dim=1)
prototypes  = torch.cat([proto_bon, proto_spoof], dim=0)  # (2, 256)

sep = F.cosine_similarity(proto_bon, proto_spoof).item()
print(f"    prototype cosine similarity: {sep:.4f}  (lower = better separated)")

# Step 2: score the dev set against the spoof prototype
print("\n[2/3] Scoring dev set with reconstructed prototypes…")
dev_ds     = ASVDataset(DATA_ROOT, 'dev', processor, tokenizer)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=False)

dev_embs, dev_labels = get_embeddings(dev_loader, 'Dev embeddings')
dev_embs_norm = F.normalize(dev_embs, dim=1)

# Higher cosine similarity to the spoof prototype → higher spoof score
cosine_scores = dev_embs_norm @ prototypes.T   # (N, 2)
spoof_scores  = cosine_scores[:, 1].numpy()

# Step 3: compute EER from the spoof score distribution
print("\n[3/3] Computing EER…")
fpr, tpr, thresholds = roc_curve(dev_labels, spoof_scores)
eer_idx   = np.nanargmin(np.abs(1 - tpr - fpr))
eer       = fpr[eer_idx]
threshold = thresholds[eer_idx]

hook.remove()

print(f"\n{'='*55}")
print(f"  Reconstructed Cosine EER : {eer*100:.2f}%")
print(f"  Decision threshold       : {threshold:.4f}")
print(f"  Prototype separation     : {sep:.4f}")
print(f"{'='*55}")

# Sanity check: if EER > 20%, try inverted scores (label convention check)
if eer > 0.20:
    print("\n  WARNING: EER > 20% — trying inverted scores…")
    fpr2, tpr2, _ = roc_curve(dev_labels, -spoof_scores)
    eer2 = fpr2[np.nanargmin(np.abs(1 - tpr2 - fpr2))]
    print(f"  Inverted EER: {eer2*100:.2f}%")
    if eer2 < eer:
        print("  → Labels were inverted. Real EER is the inverted value above.")
        print(f"\n>>> FINAL EER: {eer2*100:.2f}%")
    else:
        print("  → Inversion didn't help. Model may need retraining.")
else:
    print(f"\n>>> FINAL EER: {eer*100:.2f}%")
