"""
Rebuilds the AAM-Softmax class prototypes from the training partition, scores
the development partition by cosine similarity to the spoof prototype, and
reports the Equal Error Rate.
"""

import os

# Only use locally cached HuggingFace weights; never hit the network.
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

# How many training utterances to use when building prototypes.
# None uses the full partition.
PROTO_SAMPLES = None


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = os.environ.get("ASV_DATA_ROOT")
    assert DATA_ROOT, "Set ASV_DATA_ROOT in your .env file"
    BATCH_SIZE = 128

    print(f"Device: {DEVICE}")

    model = MultimodalVishingDetector(embed_dim=256).to(DEVICE)
    state = torch.load('src/models/best_model.pth', map_location=DEVICE)
    result = model.load_state_dict(state, strict=False)
    print(f"Loaded | unexpected: {len(result.unexpected_keys)} | "
          f"missing: {len(result.missing_keys)}")
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained(
        'facebook/wav2vec2-base',
        padding='max_length', max_length=16000 * 4, truncation=True
    )
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # The fusion layer has no public accessor, so grab its output with a hook.
    emb_cache = []

    def _hook(module, inp, out):
        emb_cache.append(out.detach().float().cpu())

    hook = model.fusion.register_forward_hook(_hook)

    def get_embeddings(loader, desc):
        emb_cache.clear()
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                iv = batch['input_values'].to(DEVICE)
                am = batch['attention_mask'].to(DEVICE)
                ti = batch['transcript_ids'].to(DEVICE)
                tm = batch['transcript_mask'].to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=DEVICE.type == 'cuda'):
                    model(iv, am, ti, tm)
                labels_all.extend(batch['labels'].cpu().numpy())
        return torch.cat(emb_cache, dim=0), np.array(labels_all)

    # Build prototypes from the training split.
    print("\nExtracting train embeddings to build prototypes...")
    train_ds = ASVDataset(DATA_ROOT, 'train', processor, tokenizer)

    if PROTO_SAMPLES is not None and PROTO_SAMPLES < len(train_ds):
        from torch.utils.data import Subset
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(train_ds), generator=gen)[:PROTO_SAMPLES].tolist()
        train_ds = Subset(train_ds, idx)
        print(f"using a {PROTO_SAMPLES}-utterance subsample for prototypes")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    train_embs, train_labels = get_embeddings(train_loader, 'Train embeddings')
    train_embs_norm = F.normalize(train_embs, dim=1)

    bonafide_mask = train_labels == 0
    spoof_mask = train_labels == 1
    print(f"bonafide samples: {bonafide_mask.sum()} | "
          f"spoof samples: {spoof_mask.sum()}")

    proto_bon = F.normalize(
        train_embs_norm[bonafide_mask].mean(dim=0, keepdim=True), dim=1)
    proto_spoof = F.normalize(
        train_embs_norm[spoof_mask].mean(dim=0, keepdim=True), dim=1)
    prototypes = torch.cat([proto_bon, proto_spoof], dim=0)

    sep = F.cosine_similarity(proto_bon, proto_spoof).item()
    print(f"prototype cosine similarity: {sep:.4f}  (lower = better separated)")

    # Score the dev split against the spoof prototype.
    print("\nScoring dev set with reconstructed prototypes...")
    dev_ds = ASVDataset(DATA_ROOT, 'dev', processor, tokenizer)
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    dev_embs, dev_labels = get_embeddings(dev_loader, 'Dev embeddings')
    dev_embs_norm = F.normalize(dev_embs, dim=1)
    spoof_scores = (dev_embs_norm @ prototypes.T)[:, 1].numpy()

    hook.remove()

    fpr, tpr, thresholds = roc_curve(dev_labels, spoof_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    threshold = thresholds[eer_idx]

    print(f"\n{'='*55}")
    print(f"Reconstructed Cosine EER : {eer*100:.2f}%")
    print(f"Decision threshold       : {threshold:.4f}")
    print(f"Prototype separation     : {sep:.4f}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
