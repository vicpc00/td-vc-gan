import functools

import torch
import torch.nn.functional as F
import torchaudio

#Parallel wavegan version of the multiscale spec loss
def multiscale_spec_loss_pwg(signal, ref, fft_sizes, spectype = 'both', return_separated = False):
    
    losses = []
    
    for fft_size in fft_sizes:
        spec_sig = torch.stft(signal, n_fft = fft_size, hop_length = fft_size//4).abs()
        spec_ref = torch.stft(ref, n_fft = fft_size, hop_length = fft_size//4).abs()
        
        loss_lin = torch.mean(torch.pow(spec_sig - spec_ref.detach(),2), dim = (-1, -2)) / \
                   torch.mean(torch.pow(spec_sig,2), dim = (-1, -2)) #Frobenius norm
                   
        loss_log = torch.mean(torch.abs(torch.log(spec_sig) - torch.log(spec_ref.detach())), dim = (-1, -2)) / \
                   (spec_sig.shape[-1]*spec_sig.shape[-2]) #L1 norm
                   
        losses.append(loss_lin + loss_log)
        
        if return_separated:
            return torch.sum(losses), losses
        return torch.sum(losses)
    
@functools.lru_cache(maxsize=None)
def get_melspec_transform(sr, n_fft, n_mels, device):
    t = torchaudio.transforms.MelSpectrogram(sample_rate = sr, n_fft = n_fft, hop_length = n_fft//4, n_mels = n_mels)
    return t.to(device)
    
def multiscale_spec_loss(signal, ref, fft_sizes, spectype = 'both', return_separated = False, norm_p = 1):
    
    losses = []
    
    for fft_size in fft_sizes:
        melspec_transf = get_melspec_transform(16000, fft_size, 80, signal.device)
        spec_sig = melspec_transf(signal)
        spec_sig = torch.log(torch.clamp(spec_sig, min=1e-5))
        spec_ref = melspec_transf(ref)
        spec_ref = torch.log(torch.clamp(spec_ref, min=1e-5))
        
        if norm_p == 1:
            loss = F.l1_loss(spec_sig, spec_ref.detach())
        elif norm_p == 2:
            loss = F.rms_loss(spec_sig, spec_ref.detach())
                   
        losses.append(loss)
        
        if return_separated:
            return sum(losses), losses
        return sum(losses)

def multiscale_feat_loss(feat_sig_list, feat_ref_list, norm_p = 1):
    
    losses = []
    
    for feat_sig, feat_ref in zip(feat_sig_list, feat_ref_list):
        feat_loss = 0
        for map_sig, map_ref in zip(feat_sig, feat_ref):
            if norm_p == 1:
                feat_loss += F.l1_loss(map_sig, map_ref.detach())
            elif norm_p == 2:
                feat_loss += F.rms_loss(map_sig, map_ref.detach())
        losses.append(feat_loss)
        
    return sum(losses)

def kl_loss(m_p, logs_p, m_q = 0, logs_q = None):
    """KL(P||Q)"""
    if m_q == 0:
        m_q = torch.zeros_like(m_p)
        logs_q = torch.zeros_like(logs_p)
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q)**2)) * torch.exp(-2. * logs_q)
    kl = torch.mean(kl)
    return kl




