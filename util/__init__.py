import math
import numpy as np
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
    
def f0_to_excitation(f0, step_size, sampling_rate=16000, linear=True):
    f0 = f0[:,:,:-1] #remove last to make interpolated shape match orig signal
        
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
            
    
def eq_rms(signal, target_rms):
    rms = np.sqrt((signal**2).mean())
    gain = 10**(target_rms/20)/rms
    return signal*gain
    
def load_possible(model, state_dict):
    state_dict_old = model.state_dict()
    
    messages = {
        'matched': [],
        'mismatched_size': [],
        'unmatched_keys': [],
        'missing_keys': []
        }
    
    for param in state_dict:
        if param in state_dict_old:
            if state_dict[param].shape == state_dict_old[param].shape:
                state_dict_old[param] = state_dict[param]
                messages['matched'].append(param)
            else:
                s = [slice(0,min(state_dict_old[param].shape[i], state_dict[param].shape[i])) for i in range(state_dict[param].ndim)]
                state_dict_old[param][s] = state_dict[param][s]
                messages['mismatched_size'].append(param)
        else:
            messages['unmatched_keys'].append(param)
    for param in state_dict_old:
        if param not in state_dict:
            messages['missing_keys'].append(param)
            
    return messages

def roll_batches(input, shifts, dim):
    
    repeat = [d if i != dim else 1 for i,d in enumerate(input.shape)]
    view = [1 if i != dim else -1 for i in range(input.ndim)]
    
    idx = torch.arange(input.shape[dim], device=input.device).view(view).repeat(repeat)
    
    view = [1 if i != 0 else -1 for i in range(input.ndim)]
    
    idx = (idx - shifts.view(view)) % input.shape[dim]
    
    return torch.gather(input, dim, idx)

def kaiser_filter(L, w):
    
    n = torch.arange(-L//2, L//2+1).float()
    f = torch.sin(math.pi*w*n)/(math.pi*n + 1e-8)
    f[n.shape[0]//2] = w
    f = f*torch.kaiser_window(L+1, False, 2.5)
    f = f/torch.sum(f)
    f = f.view(1,1,-1)
    
    return f

