import torch
import torch.nn.functional as F

def multiscale_spec_loss(signal, ref, fft_sizes, spectype = 'both', return_separated = False):
    
    losses = []
    
    for fft_size in fft_sizes:
        spec_sig = torch.stft(signal, n_fft = fft_size, hop_length = fft_size//4).abs()
        spec_ref = torch.stft(ref, n_fft = fft_size, hop_length = fft_size//4).abs()
        
        loss_lin = torch.mean(torch.pow(spec_sig - spec_ref.detach(),2), dim = (-1, -2)) / \
                   torch.mean(torch.pow(spec_sig,2), dim = (-1, -2)) 
                   
        loss_log = torch.mean(torch.abs(torch.log(spec_sig) - torch.log(spec_ref.detach())), dim = (-1, -2)) / \
                   (spec_sig.shape[-1]*spec_sig.shape[-2])
                   
        losses.append(loss_lin + loss_log)
        
        if return_separated:
            return torch.sum(losses), losses
        return torch.sum(losses)

def multiscale_feat_loss(feat_sig_list, feat_ref_list, norm_p = 1):
    
    losses = []
    
    for feat_sig, feat_ref in zip(feat_sig_list, feat_ref_list):
        feat_loss = 0
        for map_rec, map_real in zip(feat_sig, feat_ref):
            if norm_p == 1:
                feat_loss += F.l1_loss(feat_sig - feat_ref.detach())
            elif norm_p == 2:
                feat_loss += F.rms_loss(feat_sig - feat_ref.detach())
        losses.append(feat_loss)
        
    return torch.sum(losses)




