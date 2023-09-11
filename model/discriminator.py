import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.filtered_lrelu import FilteredLReLU
from util.dsp import kaiser_filter

class Discriminator(nn.Module):
    def __init__(self, num_classes, num_layers, num_channels_base, num_channel_mult=4, downsampling_factor=4, conditional_dim=32, conditional='both'):
        super().__init__()
        
        normalization = nn.utils.weight_norm
        leaky_relu_slope = 0.2
        num_channel_max = 1024
        
        self.discriminator = nn.ModuleList()
        
        self.discriminator += [nn.Sequential(normalization(nn.Conv1d(1,num_channels_base,
                                                                     kernel_size=15,
                                                                     padding=7,padding_mode='reflect')),
                                             FilteredLReLU(leaky_relu_slope))]
        nf = num_channels_base
        for i in range(num_layers):
            nf_prev = nf
            nf = min(nf*num_channel_mult, num_channel_max)
            
            self.discriminator += [nn.Sequential(normalization(nn.Conv1d(nf_prev,nf,
                                                                         kernel_size=downsampling_factor*10+1,
                                                                         stride=1,
                                                                         padding=downsampling_factor*5,
                                                                         groups=nf_prev)),
                                                 FilteredLReLU(leaky_relu_slope,
                                                               up_factor=2, dn_factor=2*downsampling_factor, 
                                                               up_fc=0.5, dn_fc=1/(2*downsampling_factor)))]
        self.discriminator += [nn.Sequential(normalization(nn.Conv1d(nf,nf,
                                                                     kernel_size=5,
                                                                     padding=2)),
                                             FilteredLReLU(leaky_relu_slope))]
        
        self.output = normalization(nn.Conv1d(nf, num_classes, kernel_size=3, stride=1, padding=1, bias=False))


        
    def forward(self, x, label_tgt):
        
        features = []
        for layer in self.discriminator:
            x = layer(x)
            features.append(x)
            
        x = self.output(x)
        
        idx = label_tgt.view(-1,1,1).expand(-1,1,x.shape[2])
        
        out = x.gather(dim = 1, index = idx)
        
        return out, features
    
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_disc, num_classes, num_layers, num_channels_base, num_channel_mult=4, downsampling_factor=4, conditional_dim=32, conditional='both'):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for i in range(num_disc):
            self.discriminators += [Discriminator(num_classes, num_layers, num_channels_base, num_channel_mult, downsampling_factor, conditional_dim, conditional)]
        
        #self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        
        #Kaiser filter
        L = 33
        f = kaiser_filter(L, 0.5, 2.5)
        f = f.view(1,1,-1)
        self.L = L
        
        self.register_buffer('down_filter', f, persistent=False)

        
    def forward(self,x, label_tgt):
        ret = []
        
        for disc in self.discriminators:

            ret.append(disc(x, label_tgt))
            x = F.conv1d(x, self.down_filter, 
                         stride=2, 
                         padding=self.L-1)

            
        out, features = zip(*ret)
        return list(out), list(features)
