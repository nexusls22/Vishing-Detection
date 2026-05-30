"""
train_binary.py
Main training script for the MultimodalVishingDetector.

Training strategy:
  - Loss      : AAM-Softmax on the 256-dim fusion embedding (binary: bonafide / spoof)
  - Auxiliary : CrossEntropyLoss on the attack-type head (weight = 0 during training)
  - Optimiser : AdamW with cosine annealing LR schedule and a linear warm-up
  - Evaluation: cosine EER computed against the AAM-Softmax weight vectors
  - Checkpoints: best model AND criterion weights are saved whenever EER improves

Run from the project root:
    python src/models/train_binary.py
"""

import os
import warnings
from dotenv import load_dotenv

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve
from transformers import Wav2Vec2Processor, AutoTokenizer
from src.data.ASVDataset import ASVDataset
from src.data.collate import collate_fn
from src.models.models import MultimodalVishingDetector

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark        = True

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root    = os.environ.get("ASV_DATA_ROOT")
assert data_root, "Set ASV_DATA_ROOT in your .env file"
total_epochs = 50


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax loss (ArcFace-style).

    Trains the model to produce embeddings that are angularly separable by
    subtracting a fixed margin from the target class cosine score before
    scaling and computing cross-entropy. The margin forces the model to push
    embeddings closer to their class centre than a naive softmax would.

    The weight matrix of this module serves as the learned class prototype
    vectors and is used directly for cosine-similarity scoring at evaluation.

    Args:
        embedding_dim: dimensionality of the input embeddings
        num_classes  : number of output classes
        margin       : additive angular margin (default 0.2)
        scale        : cosine score scaling factor (default 30)
    """

    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin = margin
        self.scale  = scale

    def forward(self, embeddings, labels):
        embeddings  = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight,  p=2, dim=1)
        cos_theta   = F.linear(embeddings, weight_norm).clamp(-1, 1)

        # Apply the angular margin only to the bonafide class (index 0)
        bonafide_mask = (labels == 0)
        if bonafide_mask.any():
            cos_theta[bonafide_mask, 0] -= self.margin

        return F.cross_entropy(cos_theta * self.scale, labels)


def compute_attack_accuracy(model, dataloader, device):
    """Computes classification accuracy for the auxiliary attack-type head."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Attack accuracy', colour='green'):
            iv = batch['input_values'].to(device)
            am = batch['attention_mask'].to(device)
            ti = batch['transcript_ids'].to(device)
            tm = batch['transcript_mask'].to(device)
            ai = batch['attack_idx'].to(device)

            _, aux_logits = model(iv, am, ti, tm)
            preds = aux_logits.argmax(dim=-1)
            mask  = ai != -1
            correct += (preds[mask] == ai[mask]).sum().item()
            total   += mask.sum().item()

    return correct / total if total > 0 else 0.0


def evaluate_cosine(model, dataloader, criterion_binary, device):
    """
    Computes binary EER using cosine similarity to the AAM-Softmax class prototypes.

    Scores each sample by its cosine similarity to the bonafide weight vector
    (higher = more bonafide). This matches the angular geometry the loss optimises,
    giving a more faithful EER than softmax over raw logits would.
    """
    model.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Cosine EER', colour='yellow'):
            iv = batch['input_values'].to(device)
            am = batch['attention_mask'].to(device)
            ti = batch['transcript_ids'].to(device)
            tm = batch['transcript_mask'].to(device)

            _, _, embedding, _, _ = model(iv, am, ti, tm, return_embeddings=True)

            emb_norm    = F.normalize(embedding,           p=2, dim=1)
            weight_norm = F.normalize(criterion_binary.weight, p=2, dim=1)

            # Cosine similarity to the bonafide prototype (index 0)
            cos_sim        = torch.matmul(emb_norm, weight_norm.T)
            genuine_scores = cos_sim[:, 0].cpu().numpy()

            all_scores.extend(genuine_scores)
            all_labels.extend(batch['labels'].cpu().numpy())

    # pos_label=0 because higher cosine-to-bonafide means more genuine
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=0)
    eer          = fpr[np.nanargmin(np.abs(1 - tpr - fpr))]
    return eer


def main():
    # Use all 2580 bonafide + 2580 spoof training samples (balanced)
    balanced_ds_size = 2 * 2580

    processor      = Wav2Vec2Processor.from_pretrained(
        'facebook/wav2vec2-base', padding='max_length', max_length=16000 * 4, truncation=True
    )
    text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_ds = ASVDataset(data_root, 'train', processor, text_tokenizer, samples=balanced_ds_size)
    dev_ds   = ASVDataset(data_root, 'dev',   processor, text_tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=32, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)

    model            = MultimodalVishingDetector(embed_dim=256).to(DEVICE)
    optimizer        = torch.optim.AdamW(model.parameters(), lr=1.25e-4, weight_decay=0.01)
    criterion_binary = AAMSoftmax(embedding_dim=256, num_classes=2, margin=0.2, scale=20).to(DEVICE)
    criterion_aux    = nn.CrossEntropyLoss(ignore_index=-1)

    aux_weight          = 0.0   # auxiliary loss weight (set > 0 to enable attack-type supervision)
    scaler              = torch.cuda.amp.GradScaler()
    warmup_epochs       = 3
    scheduler           = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )

    best_cosine_eer      = float('inf')
    patience             = 10
    epochs_no_improvement = 0
    min_delta            = 1e-4

    for epoch in range(total_epochs):
        model.train()
        print("DataLoader ready. Starting epoch…")

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training', colour='blue'):
            iv = batch['input_values'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            ti = batch['transcript_ids'].to(DEVICE)
            tm = batch['transcript_mask'].to(DEVICE)
            lb = batch['labels'].to(DEVICE)
            ai = batch['attack_idx'].to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                _, aux_logits, embedding, _, _ = model(iv, am, ti, tm, return_embeddings=True)
                loss_binary = criterion_binary(embedding, lb)
                loss_aux    = criterion_aux(aux_logits, ai)
                loss        = loss_binary + aux_weight * loss_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # Linear LR warm-up for the first few epochs, then cosine annealing
        if epoch < warmup_epochs:
            lr = (epoch + 1) / warmup_epochs * 1.25e-4
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            scheduler.step()

        cosine_eer = evaluate_cosine(model, dev_loader, criterion_binary, DEVICE)
        attack_acc = compute_attack_accuracy(model, dev_loader, DEVICE)
        print(f"Epoch {epoch+1} — Cosine EER: {cosine_eer:.4f} | Attack Acc: {attack_acc:.4f}")

        if cosine_eer < best_cosine_eer - min_delta:
            best_cosine_eer = cosine_eer
            epochs_no_improvement = 0
            torch.save(model.state_dict(),            'best_model.pth')
            torch.save(criterion_binary.state_dict(), 'best_criterion.pth')
            print(f"New best model saved (EER = {best_cosine_eer:.4f})")
        else:
            epochs_no_improvement += 1
            print(f"No improvement for {epochs_no_improvement} epoch(s)")

        if epochs_no_improvement >= patience or best_cosine_eer < 0.03:
            print(f"Early stopping after epoch {epoch+1}")
            break

    print(f"Training finished. Best EER: {best_cosine_eer:.4f}")


if __name__ == '__main__':
    main()
