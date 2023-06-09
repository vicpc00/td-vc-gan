import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from model.conditional_instance_norm import ConditionalInstanceNorm

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
    
class FiLMResnetBlock(nn.Module):
    def __init__(self, n_channel, n_cond, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2, weight_norm = lambda x: x):
        super().__init__()
        self.use_scale = False
        self.conv = nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(n_channel,n_channel,
                          kernel_size=kernel_size, 
                          dilation=dilation,
                          padding=dilation,padding_mode='reflect'))
                )
        self.posconv = nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(n_channel,n_channel,
                           kernel_size=1))
                )
        self.cond = weight_norm(nn.Linear(n_cond,2*n_channel))
        self.shortcut = weight_norm(nn.Conv1d(n_channel,n_channel, kernel_size=1))
    
    def forward(self,x,c):
        h = self.conv(x)
        c = self.cond(c)
        c = c.unsqueeze(-1).expand(-1,-1,h.shape[-1])
        gamma, beta = c.chunk(2, dim=1)
        if self.use_scale:
            h = h*(1+gamma)
        h = h + beta
        
        x = self.posconv(h) + self.shortcut(x)
        
        return x

class CINResnetBlock(nn.Module):
    def __init__(self, n_channel, n_cond, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2):
        super().__init__()
        self.block = nn.ModuleList([
                ConditionalInstanceNorm(n_channel, n_cond),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Conv1d(n_channel,n_channel,
                          kernel_size=kernel_size, 
                          dilation=dilation,
                          padding=dilation,padding_mode='reflect'),
                ConditionalInstanceNorm(n_channel, n_cond),
                nn.LeakyReLU(leaky_relu_slope),
                nn.Conv1d(n_channel,n_channel,
                           kernel_size=1)]
                )
        self.shortcut = nn.Conv1d(n_channel,n_channel, kernel_size=1)
        
    def _residual(self,x,c):
        for m in self.block:
            if type(m) is ConditionalInstanceNorm:
                x = m(x,c)
            else:
                x = m(x)
        return x
    
    def forward(self,x,c):
        return self._residual(x,c) + self.shortcut(x)
        
class Encoder(nn.Module):
    def __init__(self,downsample_ratios,channel_sizes, n_res_blocks, conditional_dim = 0, embedding_dim = None, norm_layer = nn.InstanceNorm1d, weight_norm = lambda x: x):
        super().__init__()
        
        model = nn.ModuleList()
        
        self.spk_conditioning = conditional_dim > 0
        self.cin = norm_layer is ConditionalInstanceNorm
        if self.cin and not self.spk_conditioning:
            print('WARNING: Using conditional instance normalization but conditional dimension is 0')
        
        
        leaky_relu_slope = 0.2
        
        
        model += [weight_norm(nn.Conv1d(1,channel_sizes[0],
                              kernel_size=7, padding=3,
                              padding_mode='reflect'))]
    
        channel_sizes[0] += conditional_dim if not self.cin else 0    
    
        for i,r in enumerate(downsample_ratios):
            model += [norm_layer(channel_sizes[i]) if not self.cin else norm_layer(channel_sizes[i], conditional_dim),
                      nn.LeakyReLU(leaky_relu_slope),
                      weight_norm(nn.Conv1d(channel_sizes[i], channel_sizes[i+1],
                                         kernel_size = 2*r,
                                         stride = r,
                                         padding=r // 2 + r % 2,))]
            for j in range(n_res_blocks): #wavenet resblocks
                if not self.cin:
                    model += [ResnetBlock(channel_sizes[i+1],dilation=3**j,
                                          leaky_relu_slope=leaky_relu_slope,
                                          norm_layer = norm_layer,
                                          weight_norm = weight_norm)]
                else:
                    model += [FiLMResnetBlock(channel_sizes[i+1],conditional_dim, dilation=3**j,
                                              leaky_relu_slope=leaky_relu_slope,
                                              weight_norm = weight_norm)]
                    
        model += [nn.LeakyReLU(leaky_relu_slope),
                  weight_norm(nn.Conv1d(channel_sizes[-1], channel_sizes[-1],
                                       kernel_size = 7, stride = 1, padding=3,))]
        if embedding_dim:
            model += [nn.LeakyReLU(leaky_relu_slope),
                      weight_norm(nn.Conv1d(channel_sizes[-1], embedding_dim,
                                           kernel_size = 7, stride = 1, padding=3, bias=False))]

        
        self.encoder = model
        #self.encoder = nn.Sequential(*model)
    
    def forward(self, x, c = None):
        
        if not self.cin:
            x = self.encoder[0](x) #input layer
            if self.spk_conditioning:
                c = c.unsqueeze(2).repeat(1,1,x.size(2))
                x = torch.cat([x,c],dim=1)
            for mod in self.encoder[1:]:
                x = mod(x)
        else:
            for mod in self.encoder:
                print(type(mod))
                if type(mod) in [CINResnetBlock, ConditionalInstanceNorm, FiLMResnetBlock]:
                    x = mod(x,c)
                else:
                    x = mod(x)
        x = F.normalize(x, dim=1)
        return x
        #return self.encoder(x)
          

class Decoder(nn.Module):
    def __init__(self,upsample_ratios,channel_sizes, n_res_blocks, conditional_dim = 0, embedding_dim = None, norm_layer = nn.InstanceNorm1d, weight_norm = lambda x: x):
        super().__init__()
        
        model = nn.ModuleList()
        
        self.spk_conditioning = conditional_dim > 0
        #self.cin = norm_layer is ConditionalInstanceNorm
        self.cin = True
        if self.cin and not self.spk_conditioning:
            print('WARNING: Using conditional instance normalization but conditional dimension is 0')
        channel_sizes[0] += conditional_dim if not self.cin else 0
            
        leaky_relu_slope = 0.2
        
        if embedding_dim:
            model += [nn.LeakyReLU(leaky_relu_slope),
                      weight_norm(nn.Conv1d(embedding_dim, channel_sizes[0], 
                                           kernel_size = 7, stride = 1, padding=3, bias=False))]
        
        model += [nn.LeakyReLU(leaky_relu_slope),
                  weight_norm(nn.Conv1d(channel_sizes[0], channel_sizes[0],
                                       kernel_size = 7, stride = 1, padding=3,))]
        
        
        for i,r in enumerate(upsample_ratios):
            model += [norm_layer(channel_sizes[i]) if not self.cin else norm_layer(channel_sizes[i], conditional_dim),
                      nn.LeakyReLU(leaky_relu_slope),
                      weight_norm(nn.ConvTranspose1d(channel_sizes[i], channel_sizes[i+1],
                                                     kernel_size = 2*r,
                                                     stride = r,
                                                     padding=r // 2 + r % 2, #might only work for even r
                                                     output_padding=r % 2))]
            for j in range(n_res_blocks): #wavenet resblocks
                if not self.cin:
                    model += [ResnetBlock(channel_sizes[i+1],dilation=3**j,
                                          leaky_relu_slope=leaky_relu_slope,
                                          norm_layer = norm_layer,
                                          weight_norm = weight_norm)]
                else:
                    model += [FiLMResnetBlock(channel_sizes[i+1],conditional_dim, dilation=3**j,
                                             leaky_relu_slope=leaky_relu_slope,
                                             weight_norm = weight_norm)]
        
        model += [norm_layer(channel_sizes[-1]) if not self.cin else norm_layer(channel_sizes[-1], conditional_dim),
                  nn.LeakyReLU(leaky_relu_slope),
                  weight_norm(nn.Conv1d(channel_sizes[-1],1,
                                        kernel_size=7, padding=3,
                                        padding_mode='reflect')),
                  nn.Tanh()]
        self.decoder = model
        #self.decoder = nn.Sequential(*model)
        
    def forward(self, x, c = None):
        if not self.cin:
            if self.spk_conditioning:
                c = c.unsqueeze(2).repeat(1,1,x.size(2))
                x = torch.cat([x,c],dim=1)
            for mod in self.decoder:
                x = mod(x)
        else:
            for mod in self.decoder:
                if type(mod) in [CINResnetBlock, ConditionalInstanceNorm, FiLMResnetBlock]:
                    x = mod(x,c)
                else:
                    x = mod(x)
        return x
        #return self.decoder(x)

class Generator(nn.Module):
    def __init__(self, decoder_ratios, decoder_channels, 
                 num_bottleneck_layers, num_classes, conditional_dim, content_dim = None, num_res_blocks = 3,
                 norm_layer = None, weight_norm = None, #either None, str or (str,str,str)
                 bot_cond = 'target', enc_cond = None, dec_cond = None, 
                 output_content_emb = False):
        super().__init__()
        
        self.output_content_emb = output_content_emb
        
        if type(norm_layer) is not tuple:
            nl = util.get_norm_layer(norm_layer)
            enc_norm_layer = nl
            dec_norm_layer = nl
            bot_norm_layer = nl
        else:
            bot_norm_layer = util.get_norm_layer(norm_layer[0])
            enc_norm_layer = util.get_norm_layer(norm_layer[1])
            dec_norm_layer = util.get_norm_layer(norm_layer[2])
            
            
        if type(weight_norm) is not tuple:
            nl = util.get_weight_norm(weight_norm)
            enc_weight_norm = nl
            dec_weight_norm = nl
            bot_weight_norm = nl
        else:
            bot_weight_norm = util.get_weight_norm(weight_norm[0])
            enc_weight_norm = util.get_weight_norm(weight_norm[1])
            dec_weight_norm = util.get_weight_norm(weight_norm[2])
            
            
        bot_cond_dim = conditional_dim if bot_cond == 'target' else 2*conditional_dim
        enc_cond_dim = 0 if enc_cond == None else conditional_dim
        dec_cond_dim = 0 if dec_cond == None else conditional_dim
        
        self.both_cond = bot_cond == 'both'
        
        
        self.cin = bot_norm_layer is ConditionalInstanceNorm
        self.cin = True
        
        self.decoder = Decoder(decoder_ratios, decoder_channels[:], num_res_blocks, dec_cond_dim, content_dim, dec_norm_layer, dec_weight_norm)
        self.encoder = Encoder(decoder_ratios[::-1], decoder_channels[::-1], num_res_blocks, enc_cond_dim, content_dim, enc_norm_layer, enc_weight_norm)
        
        bottleneck = nn.ModuleList()
        if not self.cin:
            bot_dim = decoder_channels[0]+bot_cond_dim
            for i in range(num_bottleneck_layers):
                bottleneck += [ResnetBlock(bot_dim, dilation=1,
                                           norm_layer=bot_norm_layer, weight_norm = bot_weight_norm)]
            bottleneck += [bot_weight_norm(nn.Conv1d(bot_dim,decoder_channels[0],kernel_size=1))]
        else:
            
            bot_dim = decoder_channels[0]
            for i in range(num_bottleneck_layers):
                bottleneck += [FiLMResnetBlock(bot_dim,bot_cond_dim, dilation=1, weight_norm = bot_weight_norm)]
        #self.bottleneck = nn.Sequential(*bottleneck)
        self.bottleneck = bottleneck

        self.embedding = nn.Linear(num_classes, conditional_dim)
        #self.embedding = nn.Embedding(num_classes, conditional_dim)
        
    def _bottleneck(self,x,c):
        #print(self.bottleneck)
        if not self.cin:
            c = c.unsqueeze(2).repeat(1,1,x.size(2))
            x = torch.cat([x,c],dim=1)
            for mod in self.bottleneck:
                x = mod(x)
        else:
            for mod in self.bottleneck:
                x = mod(x,c)
            #x = self.bottleneck(x,c)
        return x

    def forward(self,x,c_tgt, c_src = None):
        c_tgt = self.embedding(c_tgt)
        c_src = self.embedding(c_src) if c_src != None else None
        
        x = self.encoder(x,c_src)
        if self.output_content_emb:
            self.content_embedding=x
        
        if self.both_cond:
            c = torch.cat([c_src,c_tgt],dim=1)
            x = self._bottleneck(x,c)
        else:
            x = self._bottleneck(x,c_tgt)
        
        x = self.decoder(x,c_tgt)
        
#        if self.output_content_emb:
#            return x, content_emb
        
        return x
