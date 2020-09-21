import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionaInstanceNorm(nn.Module):
    def __init__(self, n_channel, n_cond):
        super().__init__()
        self.norm = nn.InstanceNorm1d(n_channel, affine=False)
        self.embedding = nn.Linear(n_cond, n_channel*2)
    def forward(self, x, c):
        h = self.embedding(c)
        h = h.unsqueeze(2)
        
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class DecoderResnetBlock(nn.Module):
    def __init__(self, n_channel, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2):
        super().__init__()
        normalization = nn.utils.weight_norm
        self.block = nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                normalization(nn.Conv1d(n_channel,n_channel,
                                        kernel_size=kernel_size, 
                                        dilation=dilation,
                                        padding=dilation,padding_mode='reflect')),
                nn.LeakyReLU(leaky_relu_slope),
                normalization(nn.Conv1d(n_channel,n_channel,kernel_size=1)))
        self.shortcut = normalization(nn.Conv1d(n_channel,n_channel,kernel_size=1))

    def forward(self,x):
        return self.block(x) + self.shortcut(x)
    
#ResNet block from StarGAn. Main diff is relu-conv-norm order
class TranformResnetBlock(nn.Module):
    def __init__(self, n_channel, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2, norm_layer=nn.InstanceNorm1d):
        super().__init__()
        self.block = nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                nn.Conv1d(n_channel,n_channel,
                          kernel_size=kernel_size, 
                          dilation=dilation,
                          padding=dilation,padding_mode='reflect'),
                norm_layer(n_channel),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Conv1d(n_channel,n_channel,
                           kernel_size=1),
                norm_layer(n_channel))
        self.shortcut = nn.Conv1d(n_channel,n_channel, kernel_size=1)
    
    def forward(self,x):
        return self.block(x) + self.shortcut(x)
    
class ResnetBlock(nn.Module):
    def __init__(self, n_channel, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2, norm_layer=nn.InstanceNorm1d, weight_norm = lambda x: x):
        super().__init__()
        self.block = nn.Sequential(
                norm_layer(n_channel),
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(n_channel,n_channel,
                                      kernel_size=kernel_size, 
                                      dilation=dilation,
                                      padding=dilation,padding_mode='reflect')),
                norm_layer(n_channel),
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(n_channel,n_channel,
                                      kernel_size=1))
                )
        self.shortcut = weight_norm(nn.Conv1d(n_channel,n_channel, kernel_size=1))
    
    def forward(self,x):
        return self.block(x) + self.shortcut(x)

class CINResnetBlock(nn.Module):
    def __init__(self, n_channel, n_cond, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2):
        super().__init__()
        self.block = nn.ModuleList([
                ConditionaInstanceNorm(n_channel, n_cond),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Conv1d(n_channel,n_channel,
                          kernel_size=kernel_size, 
                          dilation=dilation,
                          padding=dilation,padding_mode='reflect'),
                ConditionaInstanceNorm(n_channel, n_cond),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Conv1d(n_channel,n_channel,
                           kernel_size=1)]
                )
        self.shortcut = nn.Conv1d(n_channel,n_channel, kernel_size=1)
        
    def _residual(self,x,c):
        for m in self.block:
            if type(m) is ConditionaInstanceNorm:
                x = m(x,c)
            else:
                x = m(x)
        return x
    
    def forward(self,x,c):
        return self._residual(x,c) + self.shortcut(x)
        
class Encoder(nn.Module):
    def __init__(self,downsample_ratios,channel_sizes, n_res_blocks, normalization = 'weight_norm', speaker_conditioning = False):
        super().__init__()
        
        model = nn.ModuleList()
        if normalization == 'weight_norm':
            weight_norm = nn.utils.weight_norm
            norm_layer = torch.nn.Identity
        elif normalization == 'instance_norm':
            norm_layer = nn.InstanceNorm1d
            weight_norm = lambda x: x
        elif normalization == 'cond_instance_norm':
            norm_layer = ConditionaInstanceNorm
            weight_norm = lambda x: x
        leaky_relu_slope = 0.2
        
        
        model += [weight_norm(nn.Conv1d(1,channel_sizes[0],
                              kernel_size=7, padding=3,
                              padding_mode='reflect'))]
        
        for i,r in enumerate(downsample_ratios):
            model += [norm_layer(channel_sizes[0]),
                      nn.LeakyReLU(leaky_relu_slope),
                      weight_norm(nn.Conv1d(channel_sizes[i], channel_sizes[i+1],
                                         kernel_size = 2*r,
                                         stride = r,
                                         padding=r // 2 + r % 2,))]
            for j in range(n_res_blocks): #wavenet resblocks
                model += [ResnetBlock(channel_sizes[i+1],dilation=3**j,
                                      leaky_relu_slope=leaky_relu_slope,
                                      norm_layer = norm_layer,
                                      weight_norm = weight_norm)]

        
        #self.encoder = model
        self.encoder = nn.Sequential(*model)
    
    def forward(self,x):
        """
        print(x.shape)
        for model in self.encoder:
            x = model(x)
            print(x.shape)
        return x
        """
        return self.encoder(x)
          

class Decoder(nn.Module):
    def __init__(self,upsample_ratios,channel_sizes, n_res_blocks, normalization = 'weight_norm', speaker_conditioning = False):
        super().__init__()
        
        model = nn.ModuleList()
        if normalization == 'weight_norm':
            weight_norm = nn.utils.weight_norm
            norm_layer = torch.nn.Identity
        elif normalization == 'instance_norm':
            norm_layer = nn.InstanceNorm1d
            weight_norm = lambda x: x
        elif normalization == 'cond_instance_norm':
            norm_layer = ConditionaInstanceNorm
            weight_norm = lambda x: x
            
        leaky_relu_slope = 0.2
        
        for i,r in enumerate(upsample_ratios):
            model += [norm_layer(channel_sizes[i]),
                      nn.LeakyReLU(leaky_relu_slope),
                      weight_norm(nn.ConvTranspose1d(channel_sizes[i], channel_sizes[i+1],
                                                     kernel_size = 2*r,
                                                     stride = r,
                                                     padding=r // 2 + r % 2, #might only work for even r
                                                     output_padding=r % 2))]
        for j in range(n_res_blocks): #wavenet resblocks
            model += [ResnetBlock(channel_sizes[i+1],dilation=3**j,
                                  leaky_relu_slope=leaky_relu_slope,
                                  norm_layer = norm_layer,
                                  weight_norm = weight_norm)]
        
        model += [norm_layer(channel_sizes[i]),
                  nn.LeakyReLU(leaky_relu_slope),
                  weight_norm(nn.Conv1d(channel_sizes[-1],1,
                                        kernel_size=7, padding=3,
                                        padding_mode='reflect')),
                  nn.Tanh()]
        #self.decoder = model
        self.decoder = nn.Sequential(*model)
        
    def forward(self,x):
        """
        print(x.shape)
        for model in self.decoder:
            x = model(x)
            print(x.shape)
        return x
        """
        return self.decoder(x)
    
class Generator(nn.Module):
    def __init__(self, decoder_ratios, decoder_channels, num_bottleneck_layers, num_classes, conditional_dim,cond_instnorm=False):
        super().__init__()
        num_res_blocks = 3
        self.cond_instnorm = cond_instnorm
        self.decoder = Decoder(decoder_ratios, decoder_channels, num_res_blocks, 'instance_norm', False)
        self.encoder = Encoder(decoder_ratios[::-1], decoder_channels[::-1], num_res_blocks, 'instance_norm', False)
        
        
        bottleneck = nn.ModuleList()
        if not self.cond_instnorm:
            bot_dim = decoder_channels[0]+conditional_dim
            for i in range(num_bottleneck_layers):
                bottleneck += [ResnetBlock(bot_dim, dilation=1)]
            bottleneck += [nn.utils.weight_norm(nn.Conv1d(bot_dim,decoder_channels[0],kernel_size=1))]
        else:
            bot_dim = decoder_channels[0]
            for i in range(num_bottleneck_layers):
                bottleneck += [CINResnetBlock(bot_dim,conditional_dim, dilation=1)]
        #self.bottleneck = nn.Sequential(*bottleneck)
        self.bottleneck = bottleneck

        self.embedding = nn.Linear(num_classes, conditional_dim)
        #self.embedding = nn.Embedding(num_classes, conditional_dim)
        
    def _bottleneck(self,x,c):
        #print(self.bottleneck)
        if not self.cond_instnorm:
            c = c.unsqueeze(2).repeat(1,1,x.size(2))
            x = torch.cat([x,c],dim=1)
            for mod in self.bottleneck:
                x = mod(x)
        else:
            for mod in self.bottleneck:
                x = mod(x,c)
            #x = self.bottleneck(x,c)
        return x

    def forward(self,x,c):
        x = self.encoder(x)
        c = self.embedding(c)

        x = self._bottleneck(x,c)
        
        x = self.decoder(x)
        
        return x