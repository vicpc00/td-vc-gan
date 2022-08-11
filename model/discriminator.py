import torch
import torch.nn as nn
import torch.nn.functional as F

class ComponentDiscriminator(nn.Module):
    def __init__(self, num_classes, conditional_dim, layer_channels, kernel_sizes, strides, groups=1, 
                 normalization = 'weight', conv_dim = 1, conditional='target', leaky_relu_slope = 0.2):
        super().__init__()
        
        self.conditional = conditional
        
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
            conv = nn.conv2d
            
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
        self.output_adversarial = norm(conv(nc,1, kernel_size=ks, stride=1, padding=1, bias=False))
        self.output_classification = norm(conv(nc,conditional_dim, kernel_size=ks, stride=1, padding=1, bias=False))
        if conditional=='both':
            self.embedding = nn.Linear(2*num_classes, conditional_dim)
        else:
            self.embedding = nn.Linear(num_classes, conditional_dim)
        
    def forward(self,x,c_tgt, c_src=None):
        if c_src != None and self.conditional == 'both':
            c = self.embedding(torch.cat((c_tgt, c_src),dim=1))
        else:
            c = self.embedding(c_tgt)
        
        features = []
        for layer in self.discriminator:
            x = layer(x)
            features.append(x)
        out_adv = self.output_adversarial(x)
        out_cls = self.output_classification(x)
        #TODO: make choice of cost customizable
        
        out_adv = torch.flatten(out_adv, start_dim=2)
        out_cls = torch.flatten(out_cls, start_dim=2)
        
        c = c.unsqueeze(2).repeat(1,1,out_cls.size(2))
        out_cls = torch.mean(c*out_cls,dim=1).unsqueeze(1)
        
        return out_adv, out_cls, features 
    
class ScaleDiscriminator(ComponentDiscriminator):
    def __init__(self, num_classes, conditional_dim, layer_channels, kernel_size, stride, 
                 normalization = 'weight', conditional='target', leaky_relu_slope = 0.2):
        layer_channels = layer_channels + layer_channels[-1:]
        num_layers = len(layer_channels)
        kernel_sizes = [15] + [kernel_size for i in range(num_layers-2)] + [5]
        strides = [1] + [stride for i in range(num_layers-2)] + [1]
        groups = [1] + [c//4 for c in layer_channels[1:-1]] + [1]
        
        super().__init__(num_classes, conditional_dim, layer_channels, kernel_sizes, strides, groups, 
                         normalization = normalization, conv_dim = 1, conditional=conditional, leaky_relu_slope = leaky_relu_slope)
        
class PeriodDiscriminator(ComponentDiscriminator):
    def __init__(self,num_classes, conditional_dim, layer_channels, kernel_size, stride,
                 normalization = 'weight', conditional='target', leaky_relu_slope = 0.2):
        layer_channels = layer_channels + layer_channels[-1:]
        num_layers = len(layer_channels)
        strides = [stride for i in range(num_layers-1)] + [1]
        
        super().__init__(num_classes, conditional_dim, layer_channels, kernel_size, strides, groups=1, 
                         normalization = normalization, conv_dim = 2, conditional=conditional, leaky_relu_slope = leaky_relu_slope)
    
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_scales, num_classes, conditional_dim, layer_channels , kernel_size = 41, stride = 4):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            #self.discriminators += [Discriminator(num_classes, num_layers, num_channels_base, num_channel_mult, downsampling_factor, conditional_dim)]
            self.discriminators += [ScaleDiscriminator(num_classes, conditional_dim, layer_channels, kernel_size, stride)]
            
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)
        

        
    def forward(self,x,c_tgt, c_src=None):
        ret = []
        
        for disc in self.discriminators:
            ret.append(disc(x,c_tgt, c_src))
            x = self.pooling(x)
            
        if len(ret)>0:
            out_adv, out_cls, features = zip(*ret)
        else:
            out_adv, out_cls, features = [], [], []
        return list(out_adv), list(out_cls), list(features)
    
class MultiperiodDiscriminator(nn.Module):
    def __init__(self, periods, num_classes, conditional_dim, layer_channels, kernel_size = 5, stride = 3):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for p in periods:
            #self.discriminators += [Discriminator(num_classes, num_layers, num_channels_base, num_channel_mult, downsampling_factor, conditional_dim)]
            self.discriminators += [ScaleDiscriminator(num_classes, conditional_dim, layer_channels, kernel_size, stride)]

        
    def forward(self,x,c_tgt, c_src=None):
        ret = []
        
        b,c,t = x.shape
        
        for period, disc in zip(self.periods, self.discriminators):
            x_ = F.pad(x,(0, period - (t % period) ),'reflect')
            t = x_.shape[2]
            x_ = x_.view(b,c,t//period,period)
            
            ret.append(disc(x_,c_tgt, c_src))
            
        if len(ret)>0:
            out_adv, out_cls, features = zip(*ret)
        else:
            out_adv, out_cls, features = [], [], []
        return list(out_adv), list(out_cls), list(features)
        
    
class Discriminator(nn.Module):
    def __init__(self, num_scales, periods, num_classes, conditional_dim, scale_disc_config = None, period_disc_config = None):
        super().__init__()
        if scale_disc_config == None:
            self.msd = MultiscaleDiscriminator(num_scales, num_classes, conditional_dim, layer_channels = [16,64,256,1024,1024], kernel_size = 41, stride = 4)
        else:
            self.msd = MultiscaleDiscriminator(num_scales, num_classes, conditional_dim, 
                                               layer_channels = scale_disc_config.layer_channels, 
                                               kernel_size = scale_disc_config.kernel_size, 
                                               stride = scale_disc_config.stride)
        if period_disc_config == None:
            self.mpd = MultiperiodDiscriminator(periods, num_classes, conditional_dim, layer_channels = [32,128,512,1024], kernel_size = 5, stride = 3)
        else:
            self.mpd = MultiperiodDiscriminator(periods, num_classes, conditional_dim, 
                                               layer_channels = period_disc_config.layer_channels, 
                                               kernel_size = period_disc_config.kernel_size, 
                                               stride = period_disc_config.stride)
            
    def forward(self,x,c_tgt,c_src=None):
        ret = []
        
        out_adv_msd, out_cls_msd, features_msd = self.msd(x, c_tgt, c_src)
        out_adv_mpd, out_cls_mpd, features_mpd = self.msd(x, c_tgt, c_src)
        
        return out_adv_msd+out_adv_mpd, out_cls_msd+out_cls_mpd, features_msd+features_mpd
    
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

        self.output_adversarial = normalization(nn.Conv1d(nf,1, kernel_size=3, stride=1, padding=1, bias=False))
        #self.output_classification = normalization(nn.Conv1d(nf,num_classes, kernel_size=3, stride=1, padding=1, bias=False))
        self.output_classification = normalization(nn.Conv1d(nf,conditional_dim, kernel_size=3, stride=1, padding=1, bias=False))
        if conditional=='both':
            self.embedding = nn.Linear(2*num_classes, conditional_dim)
        else:
            self.embedding = nn.Linear(num_classes, conditional_dim)
        
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
        out_cls = self.output_classification(x)
        #TODO: make choice of cost customizable
        #out_cls = F.avg_pool1d(out_cls,out_cls.size(2)).squeeze()
        
        c = c.unsqueeze(2).repeat(1,1,out_cls.size(2))
        out_cls = torch.mean(c*out_cls,dim=1).unsqueeze(1)
        
        return out_adv, out_cls, features
