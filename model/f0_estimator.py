import torch
import torch.nn as nn
import torch.nn.functional as F

class F0Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        
        normalization = nn.utils.weight_norm
        leaky_relu_slope = 0.2
        
        num_layers = 3
        stride = 4
        nf = 32
        
        self.estimator = nn.ModuleList()
        
        self.estimator += [nn.Sequential(normalization(nn.Conv1d(1,nf,
                                                                     kernel_size=15,
                                                                     padding=7,padding_mode='reflect')),
                                             nn.LeakyReLU(leaky_relu_slope, inplace=True))]
        for i in range(num_layers):
            nf_prev = nf
            nf = nf*2
            self.estimator += [nn.Sequential(normalization(nn.Conv1d(nf_prev,nf,
                                                                         kernel_size=stride*10+1,
                                                                         stride=stride,
                                                                         padding=stride*5,
                                                                         groups=nf_prev)),
                                                 nn.LeakyReLU(leaky_relu_slope, inplace=True))]
        self.estimator += [nn.Sequential(normalization(nn.Conv1d(nf,nf,
                                                                     kernel_size=5,
                                                                     padding=2)),
                                             nn.LeakyReLU(leaky_relu_slope, inplace=True))]

        self.output_voiced = normalization(nn.Conv1d(nf,1, kernel_size=3, stride=1, padding=1, bias=False))
        self.output_f0 = normalization(nn.Conv1d(nf,1, kernel_size=3, stride=1, padding=1, bias=False))


    def forward(self,x):

        for layer in self.estimator:
            x = layer(x)
        out_voiced = torch.sigmoid(self.output_voiced(x))
        out_f0 = self.output_f0(x)

        return out_f0, out_voiced

