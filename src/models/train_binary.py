"""
Stage 1 training: the binary spoof/bonafide detector. AAM-Softmax on the 256-dim
fusion embedding, AdamW with warm-up then cosine annealing, evaluated by cosine
EER against the AAM-Softmax weight vectors. The best model and criterion weights
are checkpointed whenever the EER improves.
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
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = os.environ.get("ASV_DATA_ROOT")
assert data_root, "Set ASV_DATA_ROOT in your .env file"
total_epochs = 50


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (ArcFace-style). Subtracts a fixed margin
    from the target-class cosine before scaling and cross-entropy, which pushes
    embeddings tighter toward their class centre. The weight matrix doubles as
    the learned class prototypes used for cosine scoring at evaluation.
    """

    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, weight_norm).clamp(-1, 1)

        # Margin is only applied to the bonafide class (index 0).
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
            mask = ai != -1
            correct += (preds[mask] == ai[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def evaluate_cosine(model, dataloader, criterion_binary, device):
    """Binary EER from cosine similarity to the bonafide weight vector."""
    model.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Cosine EER', colour='yellow'):
            iv = batch['input_values'].to(device)
            am = batch['attention_mask'].to(device)
            ti = batch['transcript_ids'].to(device)
            tm = batch['transcript_mask'].to(device)

            _, _, embedding, _, _ = model(iv, am, ti, tm, return_embeddings=True)

            emb_norm = F.normalize(embedding, p=2, dim=1)
            weight_norm = F.normalize(criterion_binary.weight, p=2, dim=1)

            cos_sim = torch.matmul(emb_norm, weight_norm.T)
            genuine_scores = cos_sim[:, 0].cpu().numpy()

            all_scores.extend(genuine_scores)
            all_labels.extend(batch['labels'].cpu().numpy())

    # pos_label=0: higher cosine-to-bonafide means more genuine.
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=0)
    eer = fpr[np.nanargmin(np.abs(1 - tpr - fpr))]
    return eer


def main():
    # Balanced training set: 2580 bonafide + 2580 spoof.
    balanced_ds_size = 2 * 2580

    processor = Wav2Vec2Processor.from_pretrained(
        'facebook/wav2vec2-base', padding='max_length', max_length=16000 * 4, truncation=True
    )
    text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_ds = ASVDataset(data_root, 'train', processor, text_tokenizer, samples=balanced_ds_size)
    dev_ds = ASVDataset(data_root, 'dev', processor, text_tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)

    model = MultimodalVishingDetector(embed_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.25e-4, weight_decay=0.01)
    criterion_binary = AAMSoftmax(embedding_dim=256, num_classes=2, margin=0.2, scale=20).to(DEVICE)
    criterion_aux = nn.CrossEntropyLoss(ignore_index=-1)

    aux_weight = 0.0   # set > 0 to add attack-type supervision in Stage 1
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == 'cuda')
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )

    best_cosine_eer = float('inf')
    patience = 10
    epochs_no_improvement = 0
    min_delta = 1e-4

    for epoch in range(total_epochs):
        model.train()
        print("DataLoader ready. Starting epoch...")

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training', colour='blue'):
            iv = batch['input_values'].to(DEVICE)
            am = batch['attention_mask'].to(DEVICE)
            ti = batch['transcript_ids'].to(DEVICE)
            tm = batch['transcript_mask'].to(DEVICE)
            lb = batch['labels'].to(DEVICE)
            ai = batch['attack_idx'].to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
                _, aux_logits, embedding, _, _ = model(iv, am, ti, tm, return_embeddings=True)
                loss_binary = criterion_binary(embedding, lb)
                loss_aux = criterion_aux(aux_logits, ai)
                loss = loss_binary + aux_weight * loss_aux

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # Linear warm-up for the first few epochs, then cosine annealing.
        if epoch < warmup_epochs:
            lr = (epoch + 1) / warmup_epochs * 1.25e-4
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            scheduler.step()

        cosine_eer = evaluate_cosine(model, dev_loader, criterion_binary, DEVICE)
        attack_acc = compute_attack_accuracy(model, dev_loader, DEVICE)
        print(f"Epoch {epoch+1} | Cosine EER: {cosine_eer:.4f} | Attack Acc: {attack_acc:.4f}")

        if cosine_eer < best_cosine_eer - min_delta:
            best_cosine_eer = cosine_eer
            epochs_no_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
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
