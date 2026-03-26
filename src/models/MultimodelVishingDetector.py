import torch
import torch.nn as nn

from src.models.AudioEncoder import AudioEncoder
from src.models.TextEncoder import TextEncoder

# Combines audio features form pre-trained W2V2 model and text features from pre-trained DistilBERT model to perform binary classification
class MultimodalVishingDetector(nn.Module):

    def __init__(self, audio_dim=768, text_dim=768, fusion_dim=512): # audio_dim & text_dim = Feature dim output of the respective encoders / fusion_dim size of the hidden layer inside the fusion network (maybe tuned on hp fine-tuning)

        super().__init__()
        self.acoustic_encoder = AudioEncoder(freeze=True)
        self.semantic_encoder = TextEncoder(freeze=True) # Freezing to only train the fusion layer for once (to prevent overfitting and speeds up the training)
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(), # Introduces non-linearity (input, output relation can't be represented by a straight line)
            nn.Dropout(0.3), # Helps prevent overfitting and randomly zeroing 30% of activations
            nn.Linear(fusion_dim, 2)   # binary classification - Projects the hidden representation to both classes (two logits)
        ) # Small feed-forward network - Takes concatenated feats as input outputs logits for the two classes


    def forward(self, input_values, attention_mask, transcripts): # Raw audio [batch_size, seq_len], binary mask for real vs padded positions, list of stings
        acoustic_feat = self.acoustic_encoder(input_values, attention_mask) # AudioEncoder output, shape [batch_size, audio_dim]
        semantic_feat = self.semantic_encoder(transcripts) # TextEncoder output, shape [batch_size, text_dim]
        combined = torch.cat([acoustic_feat, semantic_feat], dim=1) # Concatenation of the two feat vectors along feat_dim[dim=1], shape [batch_size, audio_dim + text_dim]
        logits = self.fusion(combined) # Raw scores, shape [batch_size, 2] - Passed to the CrossEntropyLoss func
        return logits