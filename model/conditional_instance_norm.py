import torch
import torch.nn as nn

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, n_channel, n_cond, n_conf_var=0):
        super().__init__()
        self.norm = nn.InstanceNorm1d(n_channel, affine=False)
        self.embedding = nn.Linear(n_cond, n_channel*2)
        #if n_conf_var:
        self.embedding_conv = nn.Conv1d(n_cond+1, n_channel*2, kernel_size=5, padding='same')
    def forward(self, x, c):
        if len(c.shape) == 2:
            h = self.embedding(c)
            h = h.unsqueeze(2)  
        else:
            h = self.embedding_conv(c)
        
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta