import torch
import torch.nn as nn
import torch.nn.functional as F

from .grad_rev import GradRevLayer


class LatentContentClassifier(nn.Module):
    def __init__(self,num_classes,num_channels_input, num_layers=3, num_channel_mult=2, downsampling_factor=2):
        super().__init__()
        
        normalization = nn.utils.weight_norm
        leaky_relu_slope = 0.2
        
        
        self.classifier = nn.ModuleList()
        self.classifier += [GradRevLayer()]
        
        nf = num_channels_input
        for i in range(num_layers):
            nf_prev = nf
            nf = nf*num_channel_mult
            
            self.classifier += [normalization(nn.Conv1d(nf_prev,nf,
                                                        kernel_size=downsampling_factor*10+1,
                                                        stride=downsampling_factor,
                                                        padding=downsampling_factor*5)),
                                nn.LeakyReLU(leaky_relu_slope, inplace=True)]
    
        self.classifier += [normalization(nn.Conv1d(nf,nf,kernel_size=5,padding=2)),
                            nn.LeakyReLU(leaky_relu_slope, inplace=True)]
        self.classifier += [normalization(nn.Conv1d(nf,num_classes,kernel_size=3,padding=1,bias=False))]
        
    def forward(self,x):
        for layer in self.classifier:
            x = layer(x)
        out = F.avg_pool1d(x,x.size(2)).squeeze(2)
        
        return out
    
class LatentSpeakerClassifier(nn.Module):
    def __init__(self,num_classes,num_channels_input, num_layers=3, hidden_dim = 1024):
        super().__init__()
        
        normalization = nn.utils.weight_norm
        leaky_relu_slope = 0.2
        self.classifier = nn.ModuleList()
        
        for i in range(num_layers):
            
            self.classifier += [normalization(nn.Linear(num_channels_input if i == 0 else hidden_dim,hidden_dim)),
                                nn.LeakyReLU(leaky_relu_slope, inplace=True)]
            
        self.classifier += [normalization(nn.Linear(hidden_dim,num_classes,bias=False))]
        
        
    def forward(self,x):
        for layer in self.classifier:
            x = layer(x)
        
        return x