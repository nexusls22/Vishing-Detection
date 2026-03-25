#SemanticEncoder
from transformers import DistilBertModel, DistilBertTokenizer
from torch import nn

class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze DistilBERT to save memory
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, transcripts):

        # transcripts: list of strings
        inputs = self.tokenizer(transcripts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)

        # Mean pooling over token dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled