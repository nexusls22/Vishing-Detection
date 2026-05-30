"""
collate.py
Custom collate function for ASVDataset batches.

Handles three responsibilities that cannot happen inside __getitem__:
  1. Padding / trimming waveforms to a fixed length (48 000 samples = 3 s at 16 kHz)
  2. Batched GPU augmentation (gain, noise, speed, pitch) for training-time regularisation
  3. Per-sample normalisation to zero mean / unit variance, as expected by Wav2Vec2
"""

import torch
import torchaudio.functional as taF
import torch.nn.functional as nnF


def collate_fn(batch, target_sr=16000, target_len=48000, device=None):
    """
    Collates a list of ASVDataset samples into a model-ready batch.

    Args:
        batch     : list of dicts from ASVDataset.__getitem__
        target_sr : sample rate used for speed/pitch augmentation
        target_len: fixed waveform length in samples
        device    : torch device; auto-detected if None

    Returns:
        dict with keys:
          input_values: (B, T) normalised waveforms
          attention_mask: (B, T) all-ones mask (no padding after trim/pad)
          transcript_ids: (B, 128) tokenised transcripts
          transcript_mask: (B, 128) transcript attention masks
          labels: (B, -) binary labels
          attack_idx: (B, -) attack type indices (-1 for bonafide)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    waveforms       = [item['waveform'] for item in batch]
    transcript_ids  = torch.stack([item['transcript_ids']  for item in batch])
    transcript_mask = torch.stack([item['transcript_mask'] for item in batch])
    labels          = torch.stack([item['label']           for item in batch])
    attack_idxs     = torch.stack([item['attack_idx']      for item in batch])

    # Pad or trim every waveform to the fixed target length
    padded = []
    for w in waveforms:
        if w.size(0) > target_len:
            start = torch.randint(0, w.size(0) - target_len + 1, (1,)).item()
            w = w[start:start + target_len]
        elif w.size(0) < target_len:
            w = nnF.pad(w, (0, target_len - w.size(0)))
        padded.append(w)

    waveforms_batch = torch.stack(padded).to(device)  # (B, T)

    # Random gain applied to the entire batch at once
    gain = torch.empty(waveforms_batch.size(0), device=device).uniform_(0.1, 5.0)
    waveforms_batch *= gain.unsqueeze(1)

    # Additive Gaussian noise (applied with probability 0.7)
    if torch.rand(1).item() < 0.7:
        snr_db = torch.empty(waveforms_batch.size(0), device=device).uniform_(5, 25)
        noise_std = waveforms_batch.std(dim=1, keepdim=True) / (10 ** (snr_db / 20)).unsqueeze(1)
        waveforms_batch += torch.randn_like(waveforms_batch) * noise_std

    # Per-sample speed and pitch perturbation
    for i in range(waveforms_batch.size(0)):
        if torch.rand(1).item() < 0.7:
            speed = torch.empty(1, device=device).uniform_(0.8, 1.2).item()
            w_aug, _ = taF.speed(waveforms_batch[i].unsqueeze(0), target_sr, speed)
            w_aug = w_aug.squeeze(0)
            # Re-trim or re-pad after speed change alters the length
            w_aug = w_aug[:target_len] if w_aug.size(0) > target_len \
                else nnF.pad(w_aug, (0, target_len - w_aug.size(0)))
            waveforms_batch[i] = w_aug

        if torch.rand(1).item() < 0.5:
            n_steps = torch.randint(-2, 3, (1,), device=device).item()
            w_aug = taF.pitch_shift(waveforms_batch[i].unsqueeze(0), target_sr, n_steps).squeeze(0)
            w_aug = w_aug[:target_len] if w_aug.size(0) > target_len \
                else nnF.pad(w_aug, (0, target_len - w_aug.size(0)))
            waveforms_batch[i] = w_aug

    # Normalise to zero mean / unit variance per sample — required by Wav2Vec2
    mean = waveforms_batch.mean(dim=1, keepdim=True)
    std = waveforms_batch.std(dim=1, keepdim=True) + 1e-9
    input_values = (waveforms_batch - mean) / std

    # Attention mask is all-ones because every sample has the same fixed length
    attention_mask = torch.ones_like(input_values)

    return {
        'input_values':    input_values,
        'attention_mask':  attention_mask,
        'transcript_ids':  transcript_ids,
        'transcript_mask': transcript_mask,
        'labels':          labels,
        'attack_idx':      attack_idxs,
    }
