import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        
class Encoder(nn.Module):
    def __init__(self,downsample_ratios,channel_sizes, n_res_blocks):
        super().__init__()
        
        model = nn.ModuleList()
        normalization = nn.utils.weight_norm
        leaky_relu_slope = 0.2
        
        model += [normalization(nn.Conv1d(1,channel_sizes[0],
                                          kernel_size=7, padding=3,
                                          padding_mode='reflect'))]
        
        for i,r in enumerate(downsample_ratios):
            model += [nn.LeakyReLU(leaky_relu_slope),
                      normalization(nn.Conv1d(channel_sizes[i], channel_sizes[i+1],
                                              kernel_size = 2*r,
                                              stride = r,
                                              padding=r // 2 + r % 2,))]
            for j in range(n_res_blocks): #wavenet resblocks
                model += [DecoderResnetBlock(channel_sizes[i+1],dilation=3**j,
                                      leaky_relu_slope=leaky_relu_slope)]
        
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
    def __init__(self,upsample_ratios,channel_sizes, n_res_blocks):
        super().__init__()
        
        model = nn.ModuleList()
        normalization = nn.utils.weight_norm
        leaky_relu_slope = 0.2
        
        for i,r in enumerate(upsample_ratios):
            model += [nn.LeakyReLU(leaky_relu_slope),
                      normalization(nn.ConvTranspose1d(channel_sizes[i], channel_sizes[i+1],
                                                     kernel_size = 2*r,
                                                     stride = r,
                                                     padding=r // 2 + r % 2, #might only work for even r
                                                     output_padding=r % 2))]
            for j in range(n_res_blocks): #wavenet resblocks
                model += [DecoderResnetBlock(channel_sizes[i+1],dilation=3**j,
                                      leaky_relu_slope=leaky_relu_slope)]
        
        model += [nn.LeakyReLU(leaky_relu_slope),
                  normalization(nn.Conv1d(channel_sizes[-1],1,
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
    def __init__(self, decoder_ratios, decoder_channels, num_bottleneck_layers, num_classes, conditional_dim):
        super().__init__()
        num_res_blocks = 3
        self.decoder = Decoder(decoder_ratios, decoder_channels, num_res_blocks)
        self.encoder = Encoder(decoder_ratios[::-1], decoder_channels[::-1], num_res_blocks)
        
        bot_dim = decoder_channels[0]+conditional_dim
        bottleneck = nn.ModuleList()
        
        for i in range(num_bottleneck_layers):
            bottleneck += [TranformResnetBlock(bot_dim, dilation=1)]
        bottleneck += [nn.utils.weight_norm(nn.Conv1d(bot_dim,decoder_channels[0],kernel_size=1))]
        self.bottleneck = nn.Sequential(*bottleneck)
        
        self.embedding = nn.Linear(num_classes, conditional_dim)
        #self.embedding = nn.Embedding(num_classes, conditional_dim)
        
    def forward(self,x,c):
        x = self.encoder(x)
        c = self.embedding(c)

        c = c.unsqueeze(2).repeat(1,1,x.size(2))
        x = torch.cat([x,c],dim=1)
        
        x = self.bottleneck(x)
        
        x = self.decoder(x)
        
        return x