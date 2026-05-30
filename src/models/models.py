import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, DistilBertModel, DistilBertTokenizer


class AudioEncoder(nn.Module): # Base class for nn (automatic param registration, computation of gradients for params etc.)
    """
    Class AudioEncoder
    Custom encoder, which converts audio into embeddings.
    Parameters:
        model_name: str = 'facebook/wav2vec2-base'
        freeze: bool = True
    Returns:
        pooled: torch.Tensor = Tensor of shape (batch_size, hidden_size)
    """

    def __init__(self, model_name='facebook/wav2vec2-base', freeze = True, unfreeze_last_n = 0): # True = freezes all W2V2 params - Not updated during training / False unfreezes the params of the model, will be fine-tuned to the task with optimizer.step()

        super().__init__() # initializes base nn.Module to set up internal structures
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True)
        if freeze:
            for param in self.wav2vec2.parameters(): # Iteration over all params (weights and biases) of the model
                param.requires_grad = False # Params will not update if the model is frozen

            if unfreeze_last_n > 0:
                for layer in self.wav2vec2.encoder.layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, input_values, attention_mask):

        # Forward pass of W2V2 - a Returned object contains the last hidden state = Tensor of shape (batch_size, seq_len, hidden_size) --> Hidden states of the transformer for every step (W2V2-base = 768)
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # Mean pooling over time dimension
        pooled = outputs.last_hidden_state.mean(dim=1)  # Taking the mean across dim=1 will reduce the seq dim to a single vector per sample (tensor [batch_size, hidden_size])

        return pooled


class TextEncoder(nn.Module):
    """
    Class TextEncoder
    Custom encoder, which converts text transcripts into embeddings.
    Parameters:
        freeze: bool = True
    Returns:
        pooled: torch.Tensor = Tensor of shape (batch_size, hidden_size)
    """

    def __init__(self, freeze = True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', weights_only = True) # Loads tokenizer (text into input IDs and attention masks)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased', weights_only = True) # Loads pretrained model, outputs the last hidden states of all tokens

        # Freeze just like in AudioEncoder
        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.text_encoder(input_ids, attention_mask) # Call the model and passes dict of inputs in / ** dict unpacking in python = Takes dict and turns it into keyword arguments ()
        pooled = outputs.last_hidden_state.mean(dim = 1) # Mean pooling over token dimension

        return pooled


class MultimodalVishingDetector(nn.Module):
    """
    Class MultimodalVishingDetector
    Custom model which combines audio features from a pre-trained W2V2 model and text features from a pre-trained DistilBERT model to perform binary classification
    Parameters:
        audio_dim: int = 768
        text_dim: int = 768
        fusion_dim: int = 512
    Returns:
        logits: torch.Tensor = Tensor of shape (batch_size, 2)
    """

    def __init__(self, audio_dim = 768, text_dim = 768, fusion_dim = 512, embed_dim = 256, num_attack_classes = 19): # audio_dim & text_dim = Feature dim output of the respective encoders / fusion_dim size of the hidden layer inside the fusion network (maybe tuned on hp fine-tuning)

        super().__init__()
        self.acoustic_encoder = AudioEncoder(freeze = True, unfreeze_last_n = 4)
        self.semantic_encoder = TextEncoder(freeze = True)
        self.fusion = nn.Sequential( # Small feed-forward network - Takes concatenated feats as input outputs logits for the two classes
            nn.Linear(audio_dim + text_dim, fusion_dim),
            nn.ReLU(), # Introduces non-linearity (a straight line can't represent input, output relation)
            nn.Dropout(0.3), # Helps prevent overfitting and randomly zeroing 30% of activations
            nn.Linear(fusion_dim, embed_dim)   # binary classification - Projects the hidden representation to both classes (two logits)
        )

        # Binary classification head (takes the embedding as input)
        self.binary_head = nn.Linear(embed_dim, 2)

        # Auxiliary classifier (attack type) can still use combined features
        self.aux_classifier = nn.Linear(audio_dim + text_dim, num_attack_classes) # Auxiliary head - separate linear layer on the combined features


    def forward(self, input_values, audio_attention_mask, transcript_ids, transcript_mask, return_embeddings = False): # transcript_ids, transcripts_mask): # Raw audio [batch_size, seq_len], binary mask for real vs. padded positions, list of stings
        acoustic_feat = self.acoustic_encoder(input_values, audio_attention_mask) # AudioEncoder output, shape [batch_size, audio_dim]
        semantic_feat = self.semantic_encoder(transcript_ids, transcript_mask) # TextEncoder output, shape [batch_size, text_dim]
        combined = torch.cat([acoustic_feat, semantic_feat], dim = 1) # Concatenation of the two feat vectors along feat_dim[dim=1], shape [batch_size, audio_dim + text_dim]

        # Produces embedding for margin loss
        embedding = self.fusion(combined) # Shape [batch_size, embed_dim]

        # Produces logits for binary classification (for evaluation and early stopping)
        binary_logits = self.binary_head(embedding) # Raw scores, shape [batch_size, 2] - Passed to the CrossEntropyLoss func (batch, 2)

        # Audio logits
        aux_logits = self.aux_classifier(combined) # Shape [batch_size, num_attack_classes]

        if return_embeddings:
            return binary_logits, aux_logits, embedding, acoustic_feat, semantic_feat
        else:
            return binary_logits, aux_logits