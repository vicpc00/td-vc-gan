
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from model.conditional_instance_norm import ConditionalInstanceNorm

from model.ssl_encoder import SSLEncoder

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
        #self.shortcut = weight_norm(nn.Conv1d(n_channel,n_channel, kernel_size=1))
        self.shortcut = nn.Identity()
    
    def forward(self,x):
        return self.block(x) + self.shortcut(x)
    
class FiLMResnetBlock(nn.Module):
    def __init__(self, n_channel, n_cond_const, n_cond_var = 0, dilation=1, kernel_size = 3, leaky_relu_slope = 0.2, weight_norm = lambda x: x):
        super().__init__()
        self.use_scale = True
        self.conv = nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(n_channel,n_channel,
                          kernel_size=kernel_size, 
                          dilation=dilation,
                          padding=(kernel_size*dilation - dilation)//2,
                          padding_mode='reflect'))
                )
        self.posconv = nn.Sequential(
                nn.LeakyReLU(leaky_relu_slope),
                weight_norm(nn.Conv1d(n_channel,n_channel,
                           kernel_size=1))
                )
        if n_cond_const or n_cond_var:
            self.cond_var = nn.Sequential(
                    weight_norm(nn.Conv1d(n_cond_const+n_cond_var, n_cond_const+n_cond_var, 
                                          kernel_size=3, padding='same')),
                    nn.LeakyReLU(leaky_relu_slope),
                    weight_norm(nn.Conv1d(n_cond_const+n_cond_var, n_channel*2, 
                                          kernel_size=3, padding='same')))
        #self.shortcut = weight_norm(nn.Conv1d(n_channel,n_channel, kernel_size=1))
        self.shortcut = nn.Identity()
    
    def forward(self,x,c = None):
        h = self.conv(x)
        if c is not None:
            if c.ndim == 2:
                c = self.cond(c)
                c = c.unsqueeze(-1).expand(-1,-1,h.shape[-1])
            else:
                c = self.cond_var(c)
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
                          padding=(kernel_size*dilation - dilation)//2, padding_mode='reflect'),
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
    
class ExciteDownsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor, n_layers = 2, kernel_size = 5, leaky_relu_slope = 0.2, weight_norm = lambda x: x):
        super().__init__()
        self.scale_factor = scale_factor
        self.block = nn.ModuleList()
        self.block += [weight_norm(nn.Conv1d(in_channel, out_channel,
                                             kernel_size = 2*scale_factor,
                                             stride = scale_factor,
                                             padding = scale_factor // 2,))]
        for _ in range(n_layers):
            self.block += [nn.LeakyReLU(leaky_relu_slope),
                           weight_norm(nn.Conv1d(out_channel, out_channel,
                                                 kernel_size = kernel_size,
                                                 stride = 1, padding = 'same',))
                ]
        
        self.shortcut = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        f = util.kaiser_filter(16*scale_factor, 1/scale_factor)
        f = f.expand(out_channel, 1, -1)
        self.register_buffer(f'shortcut_filter', f, persistent=False)
    
    def forward(self, x):

        x_sh = self.shortcut(x)
        x_sh = F.conv1d(x_sh, self.shortcut_filter, 
                        stride=self.scale_factor, 
                        padding=8*self.scale_factor, 
                        groups=x_sh.shape[1])

        for mod in self.block:
            x = mod(x)
        
        return x + x_sh
 
class MRFBlock(nn.Module): #Multi-Receptive Field Fusion from HiFiGAN
    def __init__(self, n_channel, n_cond_const = 0, n_cond_var = 0, dilations=[1, 3, 5], kernel_sizes = [3, 7, 11], leaky_relu_slope = 0.2, weight_norm = lambda x: x):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleList() for i in range(len(kernel_sizes))])
        self.has_cond = n_cond_const > 0 or n_cond_var > 0
        for i, kernel_size in enumerate(kernel_sizes):
            for dilation in dilations:
                self.blocks[i].append(FiLMResnetBlock(n_channel, n_cond_const, n_cond_var,
                                                 dilation, kernel_size,
                                                 leaky_relu_slope, weight_norm))
               
    def forward(self,x,c = None):
        y = 0
        for block in self.blocks:
            xs = x
            for mod in block:
                xs = mod(xs, c)
            y += xs
        y = y/len(self.blocks)
        return y
        
        
class Encoder(nn.Module):
    def __init__(self,downsample_ratios,channel_sizes, n_res_blocks, conditional_dim = 0, embedding_dim = None, norm_layer = nn.InstanceNorm1d, weight_norm = lambda x: x):
        super().__init__()
        
        model = nn.ModuleList()
        
        self.spk_conditioning = conditional_dim > 0
        self.cin = norm_layer is ConditionalInstanceNorm
        if self.cin and not self.spk_conditioning:
            print('WARNING: Using conditional instance normalization but conditional dimension is 0')
        
        
        leaky_relu_slope = 0.2
        resblock_dilations = [1, 3, 5]
        resblock_kernel_sizes = [3, 7, 11]
        
        
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
            model += [MRFBlock(channel_sizes[i+1], n_cond_const = 0, n_cond_var = 0,
                              dilations=resblock_dilations, kernel_sizes=resblock_kernel_sizes,
                              leaky_relu_slope=leaky_relu_slope, weight_norm = weight_norm)]
            """
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
            """
                    
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
        self.upsample_ratios = upsample_ratios
        self.upsample_idxs = []
        excite_channels = [8, 8, 8, 8, 8]
        resblock_dilations = [1, 3, 5]
        resblock_kernel_sizes = [3, 7, 11]
        self.subsample_out_layers = nn.ModuleList()
        subsample_out = [False, True, True, False]
        
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
            self.upsample_idxs.append(len(model))
            model += [MRFBlock(channel_sizes[i+1],conditional_dim, excite_channels[i+1],
                              dilations=resblock_dilations, kernel_sizes=resblock_kernel_sizes,
                              leaky_relu_slope=leaky_relu_slope, weight_norm = weight_norm)]
            """
            for j in range(n_res_blocks): #wavenet resblocks
                if not self.cin:
                    model += [ResnetBlock(channel_sizes[i+1],dilation=3**j,
                                          leaky_relu_slope=leaky_relu_slope,
                                          norm_layer = norm_layer,
                                          weight_norm = weight_norm)]
                else:
                    model += [FiLMResnetBlock(channel_sizes[i+1],conditional_dim, excite_channels[i+1], 
                                              dilation=3**j, leaky_relu_slope=leaky_relu_slope,
                                              weight_norm = weight_norm)]
            """
            if subsample_out[i]:
                out_block = nn.Sequential(nn.LeakyReLU(leaky_relu_slope),
                                           weight_norm(nn.Conv1d(channel_sizes[i+1],1,
                                                                 kernel_size=7, padding=3,
                                                                 padding_mode='reflect')),
                                           nn.Tanh())
                self.subsample_out_layers.append(out_block)
            else:
                self.subsample_out_layers.append(None)
        
        model += [norm_layer(channel_sizes[-1]) if not self.cin else norm_layer(channel_sizes[-1], conditional_dim),
                  nn.LeakyReLU(leaky_relu_slope),
                  weight_norm(nn.Conv1d(channel_sizes[-1],1,
                                        kernel_size=7, padding=3,
                                        padding_mode='reflect')),
                  nn.Tanh()]
        self.upsample_idxs.append(len(model)) #last element is size of model
        self.decoder = model
        #self.decoder = nn.Sequential(*model)
        
        
        self.excite_downsample = nn.ModuleList()
        
        
        for r, ch_in, ch_out in zip(self.upsample_ratios, excite_channels[:-1], excite_channels[1:]):
            self.excite_downsample += [ExciteDownsampleBlock(ch_in, ch_out, r,
                                                             weight_norm = weight_norm)]
            
        self.excite_downsample += [weight_norm(nn.Conv1d(1,excite_channels[0],
                                                        kernel_size=7, padding=3,
                                                        padding_mode='reflect'))]
        
    def get_scaled_conditioning(self, c):
        
        scaled_c = []
        for mod in reversed(self.excite_downsample):
            #print(c.shape)
            c = mod(c)
            scaled_c.append(c)
            #scaled_c.append(F.avg_pool1d(scaled_c[-1], kernel_size=4*r+1, stride=r, padding=r*2))
        return scaled_c
            
        
    def forward(self, x, c = None, c_var = None, out_subsample = False):
        subsample_out = []
        if not self.cin:
            if self.spk_conditioning:
                c = c.unsqueeze(2).repeat(1,1,x.size(2))
                x = torch.cat([x,c],dim=1)
            for mod in self.decoder:
                x = mod(x)
        else:
            if c_var is not None:
                curr_scale = 0
                c_var_scales = self.get_scaled_conditioning(c_var)
                c_const = c.unsqueeze(2).repeat(1,1,x.size(2))
                c = torch.cat([c_const, c_var_scales[-1]],dim=1)
            
            for i, mod in enumerate(self.decoder):
                if i == self.upsample_idxs[curr_scale]:
                    if self.subsample_out_layers[curr_scale] is not None:
                        x_ = self.subsample_out_layers[curr_scale](x)
                        subsample_out.append(x_)
                    
                    if c_var is not None:
                        c_const = c_const.repeat(1,1,self.upsample_ratios[curr_scale])
                        curr_scale += 1
                        c = torch.cat([c_const, c_var_scales[-1-curr_scale]],dim=1)
                if type(mod) in [CINResnetBlock, ConditionalInstanceNorm, FiLMResnetBlock, MRFBlock]:
                    x = mod(x,c)
                else:
                    x = mod(x)
        if out_subsample:
            return x, subsample_out
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
        #self.encoder = Encoder(decoder_ratios[::-1], decoder_channels[::-1], num_res_blocks, enc_cond_dim, content_dim, enc_norm_layer, enc_weight_norm)
        self.encoder = SSLEncoder()
        
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

    def forward(self,x,c_tgt, c_src = None, c_var = None, out_subsample = False):
        c_tgt = self.embedding(c_tgt)
        c_src = self.embedding(c_src) if c_src != None else None
        x = self.encoder(x)
        if self.output_content_emb:
            self.content_embedding=x
        
        if self.both_cond:
            c = torch.cat([c_src,c_tgt],dim=1)
            x = self._bottleneck(x,c)
        else:
            x = self._bottleneck(x,c_tgt)
        
        x = self.decoder(x,c_tgt, c_var, out_subsample = out_subsample)
        
#        if self.output_content_emb:
#            return x, content_emb
        
        return x
