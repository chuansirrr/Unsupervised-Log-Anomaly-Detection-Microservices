
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.emb = nn.Embedding(input_dim, 32)
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_ids):
        x = self.emb(token_ids)
        out, (h, c) = self.lstm(x)
        h = h[-1]
        return self.proj(h)
