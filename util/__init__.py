import torch
import torch.nn as nn
import librosa
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
    
    
mel_basis = None
hann_window = None
def mel_spectrogram(signal, n_fft, num_mels, sample_rate, win_len, hop_len, fmin = 0, fmax = None):
    global mel_basis, hann_window
    if fmax == None:
        fmax = sample_rate/2
    
    if mel_basis == None:
        mel_basis = librosa.filters.mel(sample_rate,n_fft,num_mels,fmin,fmax)
        mel_basis = torch.tensor(mel_basis).to(signal.device)
        hann_window = torch.hann_window(win_len).to(signal.device)
    #print(mel_basis)
    spec = torch.stft(signal.squeeze(1),n_fft,hop_length=hop_len, win_length=win_len,window=hann_window,
                      center=False, pad_mode='reflect',normalized=False,onesided=True,return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+1e-9)
    spec = torch.matmul(mel_basis,spec)
    spec = torch.log(torch.clamp(spec,min=1e-5))
    
    return spec

        
    
            
