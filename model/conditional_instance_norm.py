import torch
import torch.nn as nn

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, n_channel, n_cond):
        super().__init__()
        self.norm = nn.InstanceNorm1d(n_channel, affine=False)
        self.embedding = nn.Linear(n_cond, n_channel*2)
    def forward(self, x, c):
        h = self.embedding(c)
        h = h.unsqueeze(2)
        
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta