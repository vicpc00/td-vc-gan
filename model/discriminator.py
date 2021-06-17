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

        self.output_adversarial = normalization(nn.Conv1d(nf,1, kernel_size=3, stride=1, padding=1, bias=False))
        #self.output_classification = normalization(nn.Conv1d(nf,num_classes, kernel_size=3, stride=1, padding=1, bias=False))
        self.output_classification = normalization(nn.Conv1d(nf,conditional_dim, kernel_size=3, stride=1, padding=1, bias=False))
        if conditional=='both':
            self.embedding = nn.Linear(2*num_classes, nf)
        else:
            self.embedding = nn.Linear(num_classes, nf)
        
    def forward(self,x,c_tgt, c_src=None):
        if c_src != None:
            c = self.embedding(torch.cat((c_tgt, c_src),dim=1))
        else:
            c = self.embedding(c_tgt)
        
        features = []
        for layer in self.discriminator:
            x = layer(x)
            features.append(x)
        out_adv = self.output_adversarial(x)
        #out_cls = self.output_classification(x)
        out_cls = x
        #TODO: make choice of cost customizable
        #out_cls = F.avg_pool1d(out_cls,out_cls.size(2)).squeeze()
        
        c = c.unsqueeze(2).repeat(1,1,out_cls.size(2))
        out_cls = torch.mean(c*out_cls,dim=1).unsqueeze(1)
        
        return out_adv+out_cls, out_cls, features
    
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_disc, num_classes, num_layers, num_channels_base, num_channel_mult=4, downsampling_factor=4, conditional_dim=32):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for i in range(num_disc):
            self.discriminators += [Discriminator(num_classes, num_layers, num_channels_base, num_channel_mult, downsampling_factor, conditional_dim)]
        
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        

        
    def forward(self,x,c_tgt, c_src=None):
        ret = []
        
        for disc in self.discriminators:
            ret.append(disc(x,c_tgt, c_src))
            x = self.pooling(x)
            
        out_adv, out_cls, features = zip(*ret)
        return list(out_adv), list(out_cls), list(features)
