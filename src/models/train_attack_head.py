import os
from dotenv import load_dotenv

import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoTokenizer

from src.data.ASVDataset import ASVDataset
from src.data.collate import collate_fn
from src.models.models import MultimodalVishingDetector

load_dotenv()

def compute_attack_accuracy(model_acc, dataloader, device):
    model_acc.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Attack evaluation', leave=False):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            transcript_ids = batch['transcript_ids'].to(device)
            transcript_mask = batch['transcript_mask'].to(device)
            attack_indexes = batch['attack_idx'].to(device)

            _, aux_logits = model_acc(input_values, attention_mask, transcript_ids, transcript_mask)
            preds = aux_logits.argmax(dim=-1)
            mask = attack_indexes != -1
            correct += (preds[mask] == attack_indexes[mask]).sum().item()
            total += mask.sum().item()

        return correct / total if total > 0 else 0.0

if __name__ == '__main__':

    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = os.environ.get("ASV_DATA_ROOT")
    assert data_root, "Set ASV_DATA_ROOT in your .env file"
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base', padding='max_length',max_length=16000 * 4,
                                                      truncation=True)

    text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_ds = ASVDataset(data_root, 'train', processor, text_tokenizer)
    dev_ds = ASVDataset(data_root, 'dev', processor, text_tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory = False)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory = False)

    model = MultimodalVishingDetector(embed_dim= 256).to(DEVICE)
    state_dict = torch.load('data/training/saves/best_model2.pth')
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'aux_classifier' in name:
            param.requires_grad = True
            print(f'Unfrozen: {name}')

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    criterion_aux = torch.nn.CrossEntropyLoss(ignore_index=-1)
    total_epochs = 50
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Training loop
    for epoch in range(total_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} Attack training', leave=False):
            input_values = batch['input_values'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            transcript_ids = batch['transcript_ids'].to(DEVICE)
            transcript_mask = batch['transcript_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            attack_indexes = batch['attack_idx'].to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                _, aux_logits = model(input_values, attention_mask, transcript_ids, transcript_mask)
                loss = criterion_aux(aux_logits, attack_indexes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        attack_acc = compute_attack_accuracy(model, dev_loader, DEVICE)
        print(f'Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Attack Acc: {attack_acc:.4f}')

        if attack_acc > best_acc + 1e-4:
            best_acc = attack_acc
            torch.save(model.state_dict(), 'best_attack_head.pth')
            print(f'New best attack attack accuracy: {best_acc:.4f}')

        print(f'\nFinished. Best attack accuracy: {best_acc:.4f}')

