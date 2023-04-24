import functools

import torch
import torch.nn.functional as F
import torchaudio
import librosa

import util

@functools.lru_cache(maxsize=None)
def mel_basis(sr, n_fft, n_mels, fmin, fmax, device):
    mels = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)
    mels = torch.from_numpy(mels).float().to(device)
    
    return mels


def spec_to_melspec(spec, sr = 16000, n_mels=80, f_min = 0, f_max = None):
    n_fft = (spec.shape[-2]-1)*2
    
    mels = mel_basis(sr, n_fft, n_mels, f_min, f_max, spec.device)
    
    #TODO: matrix multiply mel and spec 
    
    return spec
    
def add_jitter(signal, jitter_range):
    jitter = torch.randint(-jitter_range, jitter_range+1, (signal.shape[0], ), device=signal.device)

    return util.roll_batches(signal, jitter, signal.ndim-1)        
    
    
    
