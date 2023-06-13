import torch
import torch.nn as nn
import torch.nn.functional as F

class ComponentDiscriminator(nn.Module):
    def __init__(self, num_classes, layer_channels, kernel_sizes, strides, groups=1, 
                 normalization = 'weight', conv_dim = 1, leaky_relu_slope = 0.2):
        super().__init__()
                
        num_layers = len(layer_channels)
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes for i in range(num_layers)]
        if type(strides) == int:
            strides = [strides for i in range(num_layers)]
        if type(groups) == int:
            groups = [groups for i in range(num_layers)]
        
        
        if normalization == 'weight':
            norm = nn.utils.weight_norm
        elif normalization == 'spectral':
            norm = nn.utils.spectral_norm
            
            
        if conv_dim == 1:
            conv = nn.Conv1d
        elif conv_dim == 2:
            conv = nn.Conv2d
            
        self.discriminator = nn.ModuleList()
        
        nc = 1
        for channels, kernel_size, stride, groups in zip(layer_channels, kernel_sizes, strides, groups):
            nc_prev = nc
            nc = channels
            padding = (kernel_size-1)//2
            if conv_dim == 2: 
                kernel_size = (kernel_size,1)
                padding = (padding,0)
            
            self.discriminator += [nn.Sequential(norm(
                                        conv(nc_prev, nc,
                                             kernel_size = kernel_size,
                                             stride = stride,
                                             padding = padding,
                                             groups = groups)),
                                        nn.LeakyReLU(leaky_relu_slope, inplace=True)
                                   )]
        
        ks = 3 if conv_dim == 1 else (3,1)
        
        self.output = norm(conv(nc, num_classes, kernel_size=ks, stride=1, padding=1, bias=False))

        
    def forward(self,x, label_tgt):
        
        features = []
        for layer in self.discriminator:
            x = layer(x)
            features.append(x)
            
        x = self.output(x)
        x = torch.flatten(x, start_dim=2)
        idx = label_tgt.view(-1,1,1).expand(-1,1,x.shape[2])
        out = x.gather(dim = 1, index = idx)
        
        return out, features 
    
class ScaleDiscriminator(ComponentDiscriminator):
    def __init__(self, num_classes, layer_channels, kernel_size, stride, 
                 normalization = 'weight', leaky_relu_slope = 0.2):
        layer_channels = layer_channels + layer_channels[-1:]
        num_layers = len(layer_channels)
        kernel_sizes = [15] + [kernel_size for i in range(num_layers-2)] + [5]
        strides = [1] + [stride for i in range(num_layers-2)] + [1]
        groups = [1] + [c//4 for c in layer_channels[1:-1]] + [1]
        
        super().__init__(num_classes, layer_channels, kernel_sizes, strides, groups, 
                         normalization = normalization, conv_dim = 1, leaky_relu_slope = leaky_relu_slope)
        
class PeriodDiscriminator(ComponentDiscriminator):
    def __init__(self,num_classes, layer_channels, kernel_size, stride,
                 normalization = 'weight', leaky_relu_slope = 0.2):
        layer_channels = layer_channels + layer_channels[-1:]
        num_layers = len(layer_channels)
        strides = [stride for i in range(num_layers-1)] + [1]
        
        super().__init__(num_classes, layer_channels, kernel_size, strides, groups=1, 
                         normalization = normalization, conv_dim = 2, leaky_relu_slope = leaky_relu_slope)
    
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_scales, num_classes, layer_channels , kernel_size = 41, stride = 4):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            #self.discriminators += [Discriminator(num_classes, num_layers, num_channels_base, num_channel_mult, downsampling_factor, conditional_dim)]
            self.discriminators += [ScaleDiscriminator(num_classes, layer_channels, kernel_size, stride)]
            
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        

        
    def forward(self,x, label_tgt):
        ret = []
        
        for disc in self.discriminators:
            ret.append(disc(x, label_tgt))
            x = self.pooling(x)
            
        if len(ret)>0:
            out, features = zip(*ret)
        else:
            out, features = [], []
        return list(out), list(features)
    
class MultiperiodDiscriminator(nn.Module):
    def __init__(self, periods, num_classes, layer_channels, kernel_size = 5, stride = 3):
        super().__init__()
        
        self.periods = periods
        self.discriminators = nn.ModuleList()
        for p in periods:
            #self.discriminators += [Discriminator(num_classes, num_layers, num_channels_base, num_channel_mult, downsampling_factor, conditional_dim)]
            self.discriminators += [PeriodDiscriminator(num_classes, layer_channels, kernel_size, stride)]

        
    def forward(self,x, label_tgt):
        ret = []
        
        b,c,t = x.shape
        
        for period, disc in zip(self.periods, self.discriminators):
            x_ = F.pad(x,(0, period - (t % period) ),'reflect')
            t_ = x_.shape[2]
            x_ = x_.view(b,c,t_//period,period)
            
            ret.append(disc(x_, label_tgt))
            
        if len(ret)>0:
            out, features = zip(*ret)
        else:
            out, features = [], []
        return list(out), list(features)
        
    
class Discriminator(nn.Module):
    def __init__(self, num_scales, periods, num_classes, scale_disc_config = None, period_disc_config = None):
        super().__init__()
        if scale_disc_config == None:
            self.msd = MultiscaleDiscriminator(num_scales, num_classes, layer_channels = [16,64,256,1024,1024], kernel_size = 41, stride = 4)
        else:
            self.msd = MultiscaleDiscriminator(num_scales, num_classes, 
                                               layer_channels = scale_disc_config.layer_channels, 
                                               kernel_size = scale_disc_config.kernel_size, 
                                               stride = scale_disc_config.stride)
        if period_disc_config == None:
            self.mpd = MultiperiodDiscriminator(periods, num_classes, layer_channels = [32,128,512,1024], kernel_size = 5, stride = 3)
        else:
            self.mpd = MultiperiodDiscriminator(periods, num_classes, 
                                               layer_channels = period_disc_config.layer_channels, 
                                               kernel_size = period_disc_config.kernel_size, 
                                               stride = period_disc_config.stride)
            
    def forward(self,x, label_tgt):
        
        out_adv_msd, features_msd = self.msd(x, label_tgt)
        out_adv_mpd, features_mpd = self.mpd(x, label_tgt)
                
        return out_adv_msd+out_adv_mpd, features_msd+features_mpd
    
#============================
#Old scale discriminator
class Discriminator_old(nn.Module):
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
                                                                         groups=nf_prev // 4)),
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
    
class MultiscaleDiscriminator_old(nn.Module):
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
