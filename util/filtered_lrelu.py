import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class FilteredLReLU(nn.Module):
    def __init__(self, negative_slope=0.01, up_factor=2, dn_factor=2, up_fc=0.5, dn_fc=0.5, filter_len=8, filter_beta=2.5):
        super().__init__()
        
        self.negative_slope = negative_slope #LReLU negative slope
        self.up_factor = up_factor           #Upsample factor
        self.dn_factor = dn_factor           #Downsample factor
        self.up_fc = up_fc                   #Upsample cut frequency
        self.dn_fc = dn_fc                   #Downsample cut frequency
        self.filter_len  = filter_len        #Length of up/down sample filter length
        self.filter_beta = filter_beta       #Length of up/down sample filter beta
        
        #Upsample
        L = filter_len*up_factor
        n = torch.arange(-L//2, L//2+1).float()
        f = torch.sin(math.pi*up_fc*n)/(math.pi*n + 1e-8) #sinc function
        f[n.shape[0]//2] = dn_fc #sinc[0]
        win = torch.kaiser_window(L+1, False, filter_beta)
        f = f*win
        f = f/torch.sum(f)
        self.register_buffer('up_filter', f, persistent=False)
        
        L = filter_len*dn_factor
        n = torch.arange(-L//2, L//2+1).float()
        f = torch.sin(math.pi*dn_fc*n)/(math.pi*n + 1e-8) #sinc function
        f[n.shape[0]//2] = dn_fc #sinc[0]
        win = torch.kaiser_window(L+1, False, filter_beta)
        f = f*win
        f = f/torch.sum(f)
        self.register_buffer('dn_filter', f, persistent=False)

    def forward(self,x):
        n_channels = x.shape[1]
        up_f = self.up_filter.view(1,1,-1).expand(n_channels, 1, -1)
        dn_f = self.dn_filter.view(1,1,-1).expand(n_channels, 1, -1)
        
        
        if self.up_factor > 1:
            x = self.up_factor*F.conv_transpose1d(x, up_f, 
                                                  stride=self.up_factor, 
                                                  padding=self.up_factor*self.filter_len//2, 
                                                  output_padding=self.up_factor-1, 
                                                  groups=n_channels)
            #print(x.shape)
        x = F.leaky_relu(x, self.negative_slope, inplace=True)
        if self.dn_factor > 1:
            x = F.conv1d(x, up_f, 
                         stride=self.dn_factor, 
                         padding=self.dn_factor*self.filter_len//2, 
                         groups=n_channels)
            #print(x.shape)
            
        return x
        
