
import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', out_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0,:]
        return self.proj(cls)
