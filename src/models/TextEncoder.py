from transformers import DistilBertModel, DistilBertTokenizer
from torch import nn

# Custom encoder which converts text transcripts into embeddings
class TextEncoder(nn.Module):

    def __init__(self, freeze=True):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Loads tokenizer (text into input IDs and attention masks)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased') # Loads pretrained model, outputs the last hidden states of all tokens
        # Freeze just like in AudioEncoder
        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, transcripts):

        # Tokenizes the transcripts list, returns tensors (on CPU), pads all sequences to the same length = longest in the batch --> Dict with keys 'input_ids' & 'attention_mask', cuts any sequence longer than max_length
        inputs = self.tokenizer(transcripts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        device = self.text_encoder.device # The device the model is currently on
        inputs = {key: value.to(device) for key, value in inputs.items()} # Key = input_ids, value = Tensor
        outputs = self.text_encoder(**inputs) # Call the model and passes dict of inputs in / ** dict unpacking in python = Takes dict and turns it into keyword arguments ()

        # Mean pooling over token dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled
