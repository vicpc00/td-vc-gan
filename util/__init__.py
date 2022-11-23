import torch
import torch.nn as nn
import torch.nn.functional as F
from model.conditional_instance_norm import ConditionalInstanceNorm

def get_norm_layer(norm):
    if norm == None:
        return nn.Identity
    if norm == 'instance_norm':
        return nn.InstanceNorm1d
    if norm == 'conditional_instance_norm':
        return ConditionalInstanceNorm
    
def get_weight_norm(norm):
    if norm == None:
        return lambda x: x
    if norm == 'weight_norm':
        return nn.utils.weight_norm
    
def f0_to_excitation(f0, step_size, sampling_rate=16000):
    
    linear = True
    
    sin_gain = 0.1
    noise_std = 0.003
    noise_gain = sin_gain/(3*noise_std)
    
    angular_freq = 2*torch.pi*f0/sampling_rate
    
    upsampled_freq = F.interpolate(angular_freq, scale_factor=step_size, mode='nearest')
    
    dev = f0.device
    
    if linear:
        upsampled_freq_linear = F.interpolate(angular_freq, scale_factor=step_size, mode='linear')
        upsampled_freq_mask = F.interpolate(torch.log(angular_freq), scale_factor=step_size, mode='linear') != -torch.inf
        
        upsampled_freq[upsampled_freq_mask] = upsampled_freq_linear[upsampled_freq_mask]
        
    
    instant_phase = torch.cumsum(upsampled_freq, -1)
    start_phase = torch.rand(1, device=dev) * 2 * torch.pi
    
    excitation = sin_gain*torch.sin(instant_phase+start_phase) + torch.randn(instant_phase.shape, device=dev)*noise_std
    
    unvoiced = upsampled_freq==0
    excitation[unvoiced] = torch.randn(excitation[unvoiced].shape, device=dev)*noise_std*noise_gain
    
    return excitation
            
