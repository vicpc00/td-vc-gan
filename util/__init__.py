import numpy as np
import torch
import torch.nn as nn
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

