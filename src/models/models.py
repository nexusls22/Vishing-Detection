"""
Multimodal vishing detection architecture: a Wav2Vec2 audio encoder and a
DistilBERT text encoder feeding a fusion network that outputs binary
spoof/bonafide logits plus auxiliary attack-type logits.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, DistilBertModel, DistilBertTokenizer


class AudioEncoder(nn.Module):
    """Wav2Vec2 backbone, mean-pooled over time to a 768-dim embedding."""

    def __init__(self, model_name='facebook/wav2vec2-base', freeze=True, unfreeze_last_n=0):
        super().__init__()
        # use_safetensors=False loads the .bin weights directly. Otherwise
        # transformers attempts an online safetensors conversion that spawns a
        # subprocess and crashes on Windows (spawn re-imports the entry script).
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=False)

        if freeze:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

            # fine-tune just the top few transformer layers.
            if unfreeze_last_n > 0:
                for layer in self.wav2vec2.encoder.layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, input_values, attention_mask):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # mean pool over time: (batch, seq_len, 768) to (batch, 768)
        return outputs.last_hidden_state.mean(dim=1)


class TextEncoder(nn.Module):
    """DistilBERT backbone, mean-pooled over tokens to a 768-dim embedding."""

    def __init__(self, freeze=True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids, attention_mask)
        return outputs.last_hidden_state.mean(dim=1)


class MultimodalVishingDetector(nn.Module):
    """
    Concatenates the 768-dim audio and text embeddings, projects them to a
    256-dim fusion embedding (trained with AAM-Softmax), and produces binary
    spoof/bonafide logits plus auxiliary attack-type logits.
    """

    def __init__(self, audio_dim=768, text_dim=768, fusion_dim=512,
                 embed_dim=256, num_attack_classes=19):
        super().__init__()
        self.acoustic_encoder = AudioEncoder(freeze=True, unfreeze_last_n=4)
        self.semantic_encoder = TextEncoder(freeze=True)

        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, embed_dim),
        )

        self.binary_head = nn.Linear(embed_dim, 2)

        # Runs on the raw concatenated features, not the fusion embedding, so it
        # doesn't interfere with the AAM-Softmax geometry.
        self.aux_classifier = nn.Linear(audio_dim + text_dim, num_attack_classes)

    def forward(self, input_values, audio_attention_mask, transcript_ids,
                transcript_mask, return_embeddings=False):
        acoustic_feat = self.acoustic_encoder(input_values, audio_attention_mask)
        semantic_feat = self.semantic_encoder(transcript_ids, transcript_mask)

        combined = torch.cat([acoustic_feat, semantic_feat], dim=1)

        embedding = self.fusion(combined)
        binary_logits = self.binary_head(embedding)
        aux_logits = self.aux_classifier(combined)

        if return_embeddings:
            return binary_logits, aux_logits, embedding, acoustic_feat, semantic_feat

        return binary_logits, aux_logits
