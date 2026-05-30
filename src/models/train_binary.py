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

# Get rid of the deprecation warning - maybe try update transformers, check for compatibility first.
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Environment & Device Configuration
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Automatic GPU/CPU Detection
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
#print(f"Using device: {DEVICE}")

# Data preparation
data_root = os.environ.get("ASV_DATA_ROOT")
assert data_root, "Set ASV_DATA_ROOT in your .env file"
total_epochs = 50


class AAMSoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin = 0.2, scale = 30):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p = 2, dim = 1)
        weight_norm = F.normalize(self.weight, p = 2, dim = 1)
        cos_theta = F.linear(embeddings, weight_norm).clamp(-1, 1)

        # Apply margin only to the genuine clas (label = 0) -> Subtract margin from the target logit.
        bonafide_mask = (labels == 0)
        if bonafide_mask.any():
            cos_theta[bonafide_mask, 0] = cos_theta[bonafide_mask, 0] - self.margin

        output = cos_theta * self.scale
        return F.cross_entropy(output, labels)


# Evaluation Helper Functions
def compute_attack_accuracy(model, dataloader, device):
    """
    Calculate accuracy for the audio attack-type classification task. High accuracy here indicates robust feature extraction.
    :param model: Model to use
    :param dataloader: Data loader for subset respectively
    :param device: Device for computation
    :return: Accuracy for different attack-types
    """
    model.eval() # Disables dropout and batch norm updates
    correct, total = 0, 0

    with torch.no_grad(): # prevents memory buildup and speeds up inference
        for batch in tqdm(dataloader, desc='Compute attack accuracy', colour='green'):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            transcript_ids = batch['transcript_ids'].to(device)
            transcript_mask = batch['transcript_mask'].to(device)
            attack_indexes = batch['attack_idx'].to(device)

            _, aux_logits = model(input_values, attention_mask, transcript_ids, transcript_mask) # Gets auxiliary classifier output
            predictions = aux_logits.argmax(dim=-1) # picks the class with the highest logit (raw score)
            mask = attack_indexes != -1 # Ignore bonafide and unknown attacks
            correct += (predictions[mask] == attack_indexes[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0 # Gives the fraction of correctly identified attack types

# def evaluate_model(model, dataloader, device):
#     """
#     Compute EER for the binary vishing detection task.
#     :param model: Model to use
#     :param dataloader: Data loader for subset respectively
#     :param device: Device for computation
#     :return: Equal Error Rate
#     """
#     model.eval()
#     all_labels = []
#     all_probs = []
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc='Evaluating', colour='yellow'): # Send the batch to the device
#             input_values = batch['input_values'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             transcript_ids = batch['transcript_ids'].to(device)
#             transcript_mask = batch['transcript_mask'].to(device)
#             labels = batch['labels'].to(device)
#
#             binary_logits, _ = model(input_values, attention_mask, transcript_ids, transcript_mask) # Gets only the binary classification output
#
#             if torch.isnan(binary_logits).any():
#                 print('Nan detected')
#                 print('input_values: ', input_values)
#                 print('attention_mask: ', attention_mask)
#                 continue
#
#             assert torch.isfinite(input_values).all(), 'input_values contain inf or nan'
#             assert attention_mask.sum() > 0, 'empty attention mask'
#
#             probs = torch.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()
#             all_probs.extend(probs) # Appends probabilities and true labels
#             all_labels.extend(labels.cpu().numpy())
#
#     fpr, tpr, _ = roc_curve(all_labels, all_probs) # Compute Receiver operation characteristics
#     eer = fpr[np.nanargmin(np.abs(1 - tpr - fpr))] # argmin returns indices of the minimum values along an axis
#     return eer

def evaluate_cosine(model, dataloader, criterion_binary, device):
    """
    Compute EER using cosine similarity to AAM‑Softmax prototypes.
    This matches the training.
    """
    model.eval()
    all_labels = []
    all_scores = []  # cosine similarity to genuine class prototype

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Cosine Eval', colour='yellow'):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            transcript_ids = batch['transcript_ids'].to(device)
            transcript_mask = batch['transcript_mask'].to(device)
            labels = batch['labels'].to(device)

            _, _, embedding, _, _ = model(
                input_values, attention_mask, transcript_ids, transcript_mask,
                return_embeddings=True
            )

            # Normalize embeddings and prototype vectors
            embedding_norm = F.normalize(embedding, p=2, dim=1)
            weight_norm = F.normalize(criterion_binary.weight, p=2, dim=1)

            # Cosine similarity to genuine class (index 0)
            cos_sim = torch.matmul(embedding_norm, weight_norm.T)  # (batch, 2)
            genuine_scores = cos_sim[:, 0].cpu().numpy()   # higher = more genuine

            all_scores.extend(genuine_scores)
            all_labels.extend(labels.cpu().numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=0)  # genuine is positive
    eer = fpr[np.nanargmin(np.abs(1 - tpr - fpr))]
    return eer

# Training Routine
def main():

    # Balance dataset size to avoid class imbalance
    balanced_ds_size = 2 * 2580

    # Load Tokenizer for the text branch with DistilBERT
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base', padding='max_length',max_length=16000 * 4, # 4 seconds = 64k
                                                  truncation=True)
    text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Create train and validation datasets
    train_ds = ASVDataset(data_root, 'train', processor, text_tokenizer, samples=balanced_ds_size)
    dev_ds = ASVDataset(data_root, 'dev', processor, text_tokenizer)

    # DataLoaders with data loading
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory = False)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory = False)

    # Instantiate the multimodel model and move it to the selected device
    model = MultimodalVishingDetector(embed_dim= 256).to(DEVICE)
    # model = torch.compile(model, mode='reduce-overhead')

    # Optimizer and loss functions
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.25e-4, weight_decay=0.01)
    criterion_binary = AAMSoftmax(embedding_dim = 256, num_classes = 2, margin = 0.2, scale = 20).to(DEVICE)
    criterion_aux = nn.CrossEntropyLoss(ignore_index=-1) # ignore bonafide and unknown attacks

    # Hyperparameter for multi-task learning
    aux_weight = 0.0 # Weight for audio attack classification loss
    lambda_consistency = 0.1 # Weight for consistency loss (aligns audio-text embeddings)

    # Mixed precision (Faster training, Lower memory usage)ser
    scaler = torch.cuda.amp.GradScaler()

    # Learning rate scheduler with warm-up
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

    # Early stopping parameters
    best_eer = float('inf')
    best_cosine_eer = float('inf')
    best_attack_acc = 0.0
    patience = 10
    epochs_no_improvement = 0
    min_delta = 1e-4

    # Training Loop
    for epoch in range(total_epochs):
        model.train()
        print("DataLoader ready. Starting epoch...")
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Training', colour='blue'):
            # Move batch to the device
            input_values = batch['input_values'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            transcript_ids = batch['transcript_ids'].to(DEVICE)
            transcript_mask = batch['transcript_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            attack_indexes = batch['attack_idx'].to(DEVICE)

            # Set gradients to zero before each iteration
            optimizer.zero_grad()

            # Forward pass inside autocast context
            with torch.amp.autocast(device_type='cuda'):
                binary_logits, aux_logits, embedding, acoustic_feat, semantic_feat = model(
                    input_values, attention_mask, transcript_ids, transcript_mask, return_embeddings=True
                )
                loss_binary = criterion_binary(embedding, labels)
                loss_aux = criterion_aux(aux_logits, attack_indexes)
                loss = loss_binary + aux_weight * loss_aux

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Unscale gradients and clip them
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step and scaler update
            scaler.step(optimizer)
            scaler.update()

        # Learning rate & Warmup
        if epoch < warmup_epochs:
            # Linear warm-up for the first few epochs
            lr = (epoch + 1) / warmup_epochs * 1.25e-4
            aux_weight = 0.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # aux_weight = 0.3
            scheduler.step()

        # Validation & Model Checkpointing
        # eval_eer = evaluate_model(model, dev_loader, DEVICE)
        cosine_eer = evaluate_cosine(model, dev_loader, criterion_binary, DEVICE)
        attack_acc = compute_attack_accuracy(model, dev_loader, DEVICE)
        print(f"Epoch {epoch + 1} - Validation Cosine EER: {cosine_eer:.4f} | Attack Acc: {attack_acc:.4f}")


        # Save best model based on EER improvement
        if cosine_eer < best_cosine_eer - min_delta:
            best_cosine_eer = cosine_eer
            epochs_no_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(criterion_binary.state_dict(), 'best_criterion.pth')  # saves AAMSoftmax weights
            print(f"New best model saved (EER = {best_cosine_eer:.4f})")
        else:
            epochs_no_improvement += 1
            print(f"No improvement for {epochs_no_improvement} epoch(s)")

        # Early stopping
        if epochs_no_improvement >= patience or best_cosine_eer < 0.03:
            print(f"Early stopping after epoch {epoch+1}")
            break

    print(f"Training finished. Best EER: {best_cosine_eer:.4f}")

# Execution of main
if __name__ == '__main__': # Execution of main()

    main()