
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a = nn.Parameter(torch.zeros(heads, 2*out_dim))
        self.heads = heads
        self.out_dim = out_dim

    def forward(self, x, adj):
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.out_dim)  # [N,H,F]
        # Dense attention with adjacency mask
        attn_logits = torch.einsum("nhf,mhf->nhm", h, h)  # similarity per head
        mask = torch.from_numpy(adj).to(h.device).unsqueeze(0)  # [1,N,N]
        attn = torch.softmax(attn_logits * mask, dim=-1)  # [N,H,N]
        out = torch.einsum("nhm,mhf->nhf", attn, h).mean(dim=1)  # [N,F]
        return F.elu(out)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers=2, heads=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SimpleGATLayer(in_dim, hidden_dim, heads=heads))
        for _ in range(layers-1):
            self.layers.append(SimpleGATLayer(hidden_dim, hidden_dim, heads=heads))
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return self.proj(x)
