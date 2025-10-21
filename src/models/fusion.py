
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, dims, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(sum(dims), out_dim)
        self.gate = nn.Sequential(
            nn.Linear(sum(dims), out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, len(dims)),
            nn.Sigmoid()
        )
        self.out_dim = out_dim

    def forward(self, xs):
        cat = torch.cat(xs, dim=1)
        g = self.gate(cat)
        splits = torch.split(cat, [x.size(1) for x in xs], dim=1)
        weighted = [splits[i] * g[:, i].unsqueeze(1) for i in range(len(xs))]
        fused = self.fc(torch.cat(weighted, dim=1))
        return F.normalize(fused, dim=1)
