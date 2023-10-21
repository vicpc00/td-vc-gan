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
    t = torchaudio.transforms.MelSpectrogram(sample_rate = sr, n_fft = n_fft, hop_length = n_fft//4, n_mels = n_mels, norm = 'slaney')
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

def contrastive_loss(sig_X, sig_Y, num_negatives = 100, temp=1):
    #Tensors: BxCxT
    
    def sample_negative(X, n_neg):
        
        B, C, T = X.shape
        
        with torch.no_grad():
            
            idxs = torch.randint(low = 0, high = T-1,
                                 size = (B, T, n_neg), 
                                 device=X.device) #BxTxN
            self_idxs = torch.arange(T, device=X.device).unsqueeze(-1).expand(-1, n_neg)
            idxs[idxs >= self_idxs] += 1
            
            #Preping to gather
            X = X.unsqueeze(2).expand(-1, -1, T, -1) #BxCxTxT; repeat on dim 2 to gather on dim 3
            idxs = idxs.unsqueeze(1).expand(-1, C, -1, -1) #BxCxTxN
            #print(X.shape, idxs.shape)
            
            negs = X.gather(3, idxs)
            
        return negs #BxCxTxN
    
    def compute_similarity(X, Y, negs, temp = 1):
        #X, Y: BxCxT
        #negs: BxCxTxN
        targets = torch.cat([Y.unsqueeze(-1), negs], dim=-1) #index 0 is the positive, rest is negative
        
        logits = F.cosine_similarity(X.unsqueeze(-1), targets, dim=1) #BxTxN
        logits = logits/temp
        
        return logits
    
    negs_X = sample_negative(sig_X, num_negatives)
    negs_Y = sample_negative(sig_Y, num_negatives)
    
    logits_X = compute_similarity(sig_X, sig_Y, negs_X)
    logits_Y = compute_similarity(sig_Y, sig_X, negs_Y)
    
    logits = torch.cat((logits_X, logits_Y), dim=0)
    targets = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
    #print(logits.shape, targets.shape)
    
    loss = F.cross_entropy(logits.transpose(1,2), targets)
    
    return loss
    
    
    
    
            
            
    



