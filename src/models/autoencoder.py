
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[384,96]):
        super().__init__()
        enc = []; last = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.encoder = nn.Sequential(*enc)
        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec += [nn.Linear(last, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
