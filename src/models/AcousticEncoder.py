#Acoustic Encoder
import torch.nn as nn
from transformers import Wav2Vec2Model

class AcousticEncoder(nn.Module): # Base class for nn (automatic param registration, computation of gradients for params etc.)
    def __init__(self, model_name='facebook/wav2vec2-base', freeze=True): # True freezes all params of the model out of own classifiers / False unfreezes the params of the model, will be fine-tuned to the task with optimizer.step()
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        if freeze:
            for param in self.wav2vec2.parameters(): # Iteration over all params (weights and biases) of the model
                param.requires_grad = False # Param will not update if model is frozen

    def forward(self, input_values, attention_mask):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        # Mean pooling over time dimension
        pooled = outputs.last_hidden_state.mean(dim=1)  # (batch, hidden_size)
        return pooled