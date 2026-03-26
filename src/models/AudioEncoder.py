import torch.nn as nn
from transformers import Wav2Vec2Model

# Custom encoder with registered, pretrained base model from W2V2 with possible freezing of the model
class AudioEncoder(nn.Module): # Base class for nn (automatic param registration, computation of gradients for params etc.)

    def __init__(self, model_name='facebook/wav2vec2-base', freeze=True): # True = freezes all W2V2 params - Not updated during training / False unfreezes the params of the model, will be fine-tuned to the task with optimizer.step()

        super().__init__() # initializes base nn.Module to set up internal structures
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        if freeze:
            for param in self.wav2vec2.parameters(): # Iteration over all params (weights and biases) of the model
                param.requires_grad = False # Params will not update if the model is frozen

    def forward(self, input_values, attention_mask):

        # Forward pass of W2V2 - Returned object contains last hidden state = Tensor of shape (batch_size, seq_len, hidden_size) --> Hidden states of the transformer for every step (W2V2-base = 768)
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # Mean pooling over time dimension
        pooled = outputs.last_hidden_state.mean(dim=1)  # Taking the mean across dim=1 will reduce the seq dim to a single vector per sample (tensor [batch_size, hidden_size])
        return pooled