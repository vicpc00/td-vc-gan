import math

import torch

def kaiser_filter(L, fc, beta = 2.5):
    if L % 2 == 0:
        raise Exception("Even length filter not implemented")
    L -= 1
    n = torch.arange(-L//2, L//2+1).float()
    f = torch.sin(math.pi*fc*n)/(math.pi*n + 1e-8) #sinc function
    f[n.shape[0]//2] = fc #sinc[0]
    win = torch.kaiser_window(L+1, False, beta)
    f = f*win
    f = f/torch.sum(f)
    
    return f

