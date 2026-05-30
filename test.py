# eval_final.py — put in project root
import os
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve
from transformers import Wav2Vec2Processor, AutoTokenizer

from src.data.ASVDataset import ASVDataset
from src.data.collate import collate_fn
from src.models.models import MultimodalVishingDetector

load_dotenv()

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.environ.get("ASV_DATA_ROOT")
assert DATA_ROOT, "Set ASV_DATA_ROOT in .env"

BATCH_SIZE = 128  # try 96 if you OOM, 192 if you have headroom

processor = Wav2Vec2Processor.from_pretrained(
    'facebook/wav2vec2-base',
    padding='max_length', max_length=16000 * 4, truncation=True
)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

dev_ds = ASVDataset(DATA_ROOT, 'dev', processor, tokenizer)
dev_loader = DataLoader(
    dev_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=0, pin_memory=False
)
print(f"Dev set size: {len(dev_ds)} utterances, batch size {BATCH_SIZE}")


def load_with_diagnostics(model, path):
    state = torch.load(path, map_location=DEVICE)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  MISSING ({len(result.missing_keys)}):",
              result.missing_keys[:3], "..." if len(result.missing_keys) > 3 else "")
    if result.unexpected_keys:
        print(f"  UNEXPECTED ({len(result.unexpected_keys)}):",
              result.unexpected_keys[:3], "..." if len(result.unexpected_keys) > 3 else "")


def eval_binary_eer(model, loader):
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Binary EER', colour='yellow'):
            iv = batch['input_values'].to(DEVICE, non_blocking=True)
            am = batch['attention_mask'].to(DEVICE, non_blocking=True)
            ti = batch['transcript_ids'].to(DEVICE, non_blocking=True)
            tm = batch['transcript_mask'].to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                logits, _ = model(iv, am, ti, tm)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            scores.extend(probs)
            labels.extend(batch['labels'].cpu().numpy())
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = fpr[np.nanargmin(np.abs(1 - tpr - fpr))]
    return eer


def eval_attack_acc(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Attack Acc', colour='green'):
            iv = batch['input_values'].to(DEVICE, non_blocking=True)
            am = batch['attention_mask'].to(DEVICE, non_blocking=True)
            ti = batch['transcript_ids'].to(DEVICE, non_blocking=True)
            tm = batch['transcript_mask'].to(DEVICE, non_blocking=True)
            ai = batch['attack_idx'].to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                _, aux = model(iv, am, ti, tm)
            preds = aux.argmax(dim=-1)
            mask = ai != -1
            correct += (preds[mask] == ai[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0


model = MultimodalVishingDetector(embed_dim=256).to(DEVICE)

#print("\n=== best_model.pth (best EER checkpoint) ===")
#load_with_diagnostics(model, 'src/models/best_model.pth')
#eer = eval_binary_eer(model, dev_loader)
#print(f">>> Binary EER: {eer:.4f}  ({eer*100:.2f}%)")

print("\n=== best_attack_head.pth (best attack-acc checkpoint) ===")
load_with_diagnostics(model, 'src/models/best_attack_head.pth')
acc = eval_attack_acc(model, dev_loader)
print(f">>> Attack Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

print("\nDONE")