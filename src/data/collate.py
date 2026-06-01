"""
Batch collation for ASVDataset.

Pads/trims waveforms to a fixed length, applies the training augmentation
pipeline, and normalises each waveform before it reaches Wav2Vec2.
"""

import torch
import torchaudio.functional as taF
import torch.nn.functional as nnF


def collate_fn(batch, target_sr=16000, target_len=48000, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    waveforms = [item['waveform'] for item in batch]
    transcript_ids = torch.stack([item['transcript_ids'] for item in batch])
    transcript_mask = torch.stack([item['transcript_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    attack_idxs = torch.stack([item['attack_idx'] for item in batch])

    # Fix every waveform to target_len: random crop if longer, zero-pad if shorter.
    padded = []
    for w in waveforms:
        if w.size(0) > target_len:
            start = torch.randint(0, w.size(0) - target_len + 1, (1,)).item()
            w = w[start:start + target_len]
        elif w.size(0) < target_len:
            w = nnF.pad(w, (0, target_len - w.size(0)))
        padded.append(w)

    waveforms_batch = torch.stack(padded).to(device)

    gain = torch.empty(waveforms_batch.size(0), device=device).uniform_(0.1, 5.0)
    waveforms_batch *= gain.unsqueeze(1)

    if torch.rand(1).item() < 0.7:
        snr_db = torch.empty(waveforms_batch.size(0), device=device).uniform_(5, 25)
        noise_std = waveforms_batch.std(dim=1, keepdim=True) / (10 ** (snr_db / 20)).unsqueeze(1)
        waveforms_batch += torch.randn_like(waveforms_batch) * noise_std

    for i in range(waveforms_batch.size(0)):
        if torch.rand(1).item() < 0.7:
            speed = torch.empty(1, device=device).uniform_(0.8, 1.2).item()
            w_aug, _ = taF.speed(waveforms_batch[i].unsqueeze(0), target_sr, speed)
            w_aug = w_aug.squeeze(0)
            # speed change alters length, so re-fix to target_len
            w_aug = w_aug[:target_len] if w_aug.size(0) > target_len \
                else nnF.pad(w_aug, (0, target_len - w_aug.size(0)))
            waveforms_batch[i] = w_aug

        if torch.rand(1).item() < 0.5:
            n_steps = torch.randint(-2, 3, (1,), device=device).item()
            w_aug = taF.pitch_shift(waveforms_batch[i].unsqueeze(0), target_sr, n_steps).squeeze(0)
            w_aug = w_aug[:target_len] if w_aug.size(0) > target_len \
                else nnF.pad(w_aug, (0, target_len - w_aug.size(0)))
            waveforms_batch[i] = w_aug

    mean = waveforms_batch.mean(dim=1, keepdim=True)
    std = waveforms_batch.std(dim=1, keepdim=True) + 1e-9
    input_values = (waveforms_batch - mean) / std
    attention_mask = torch.ones_like(input_values)

    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'transcript_ids': transcript_ids,
        'transcript_mask': transcript_mask,
        'labels': labels,
        'attack_idx': attack_idxs,
    }
