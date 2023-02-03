import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                             nn.LeakyReLU(leaky_relu_slope, inplace=True))]
        nf = num_channels_base
        for i in range(num_layers):
            nf_prev = nf
            nf = min(nf*num_channel_mult, num_channel_max)
            
            self.discriminator += [nn.Sequential(normalization(nn.Conv1d(nf_prev,nf,
                                                                         kernel_size=downsampling_factor*10+1,
                                                                         stride=downsampling_factor,
                                                                         padding=downsampling_factor*5,
                                                                         groups=nf_prev)),
                                                 nn.LeakyReLU(leaky_relu_slope, inplace=True))]
        self.discriminator += [nn.Sequential(normalization(nn.Conv1d(nf,nf,
                                                                     kernel_size=5,
                                                                     padding=2)),
                                             nn.LeakyReLU(leaky_relu_slope, inplace=True))]
        
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
        
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        

        
    def forward(self,x, label_tgt):
        ret = []
        
        for disc in self.discriminators:
            ret.append(disc(x, label_tgt))
            x = self.pooling(x)
            
        out, features = zip(*ret)
        return list(out), list(features)
