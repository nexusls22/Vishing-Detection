#MultimodelVishingDetector
import torch
import torch.nn as nn

from src.models.AcousticEncoder import AcousticEncoder
from src.models.TextEncoder import TextEncoder


class MultimodalVishingDetector(nn.Module):
    def __init__(self, acoustic_dim=768, semantic_dim=768, fusion_dim=512):
        super().__init__()
        self.acoustic_encoder = AcousticEncoder(freeze=False)  # or freeze=True if you prefer
        self.semantic_encoder = TextEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(acoustic_dim + semantic_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 2)   # binary classification
        )

    def forward(self, input_values, attention_mask, transcripts):
        acoustic_feat = self.acoustic_encoder(input_values, attention_mask)
        semantic_feat = self.semantic_encoder(transcripts)
        combined = torch.cat([acoustic_feat, semantic_feat], dim=1)
        logits = self.fusion(combined)
        return logits