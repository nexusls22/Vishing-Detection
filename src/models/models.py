"""
test
models.py
Defines the multimodal vishing detection architecture.

Three components are composed together:
  - AudioEncoder   : frozen Wav2Vec2 backbone that produces audio embeddings
  - TextEncoder    : frozen DistilBERT backbone that produces transcript embeddings
  - MultimodalVishingDetector : fuses both embeddings, outputs binary spoof/bonafide
                                logits and auxiliary attack-type logits
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, DistilBertModel, DistilBertTokenizer


class AudioEncoder(nn.Module):
    """
    Wraps Wav2Vec2 and reduces its sequence output to a single embedding
    via mean pooling over the time dimension.

    Args:
        model_name     : HuggingFace model identifier
        freeze         : freeze all Wav2Vec2 weights if True
        unfreeze_last_n: number of transformer layers to unfreeze from the top
                         (fine-tuned while the rest stays frozen)
    Returns:
        Tensor of shape (batch_size, 768)
    """

    def __init__(self, model_name='facebook/wav2vec2-base', freeze=True, unfreeze_last_n=0):
        super().__init__()
        # use_safetensors=False forces loading the .bin weights directly.
        # Without this, transformers tries an online safetensors auto-conversion
        # that spawns a subprocess — which crashes on Windows (spawn re-imports app.py).
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=False)

        if freeze:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

            # Selectively unfreeze the top N transformer layers for light fine-tuning
            if unfreeze_last_n > 0:
                for layer in self.wav2vec2.encoder.layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, input_values, attention_mask):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # Mean pool over the time dimension: (batch, seq_len, 768) → (batch, 768)
        return outputs.last_hidden_state.mean(dim=1)


class TextEncoder(nn.Module):
    """
    Wraps DistilBERT and reduces its token output to a single embedding
    via mean pooling over the token dimension.

    Args:
        freeze: freeze all DistilBERT weights if True
    Returns:
        Tensor of shape (batch_size, 768)
    """

    def __init__(self, freeze=True):
        super().__init__()
        self.tokenizer    = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids, attention_mask)
        # Mean pool over the token dimension: (batch, seq_len, 768) → (batch, 768)
        return outputs.last_hidden_state.mean(dim=1)


class MultimodalVishingDetector(nn.Module):
    """
    Fuses audio and text embeddings for binary vishing detection.

    Architecture:
      1. AudioEncoder  → 768-dim audio embedding
      2. TextEncoder   → 768-dim text embedding
      3. Concatenation → 1536-dim combined feature vector
      4. Fusion MLP    → 256-dim shared embedding (trained with AAM-Softmax)
      5. binary_head   → 2 logits  (bonafide / spoof)
      6. aux_classifier→ N logits  (attack type, auxiliary task)

    Args:
        audio_dim         : output dim of AudioEncoder (default 768)
        text_dim          : output dim of TextEncoder  (default 768)
        fusion_dim        : hidden size of the fusion MLP (default 512)
        embed_dim         : size of the shared embedding used for AAM-Softmax (default 256)
        num_attack_classes: number of spoof attack categories (default 19)
    """

    def __init__(self, audio_dim=768, text_dim=768, fusion_dim=512,
                 embed_dim=256, num_attack_classes=19):
        super().__init__()
        self.acoustic_encoder = AudioEncoder(freeze=True, unfreeze_last_n=4)
        self.semantic_encoder = TextEncoder(freeze=True)

        # Fusion MLP: projects concatenated features to a compact shared embedding
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, embed_dim),
        )

        # Binary classification head — operates on the shared embedding
        self.binary_head = nn.Linear(embed_dim, 2)

        # Auxiliary attack-type classifier — operates directly on the combined features
        # (kept separate from the fusion embedding to avoid interference with AAM-Softmax)
        self.aux_classifier = nn.Linear(audio_dim + text_dim, num_attack_classes)

    def forward(self, input_values, audio_attention_mask,
                transcript_ids, transcript_mask, return_embeddings=False):
        acoustic_feat = self.acoustic_encoder(input_values, audio_attention_mask)
        semantic_feat = self.semantic_encoder(transcript_ids, transcript_mask)

        # Concatenate along the feature dimension → (batch, audio_dim + text_dim)
        combined = torch.cat([acoustic_feat, semantic_feat], dim=1)

        embedding     = self.fusion(combined)
        binary_logits = self.binary_head(embedding)
        aux_logits    = self.aux_classifier(combined)

        if return_embeddings:
            return binary_logits, aux_logits, embedding, acoustic_feat, semantic_feat
        return binary_logits, aux_logits
