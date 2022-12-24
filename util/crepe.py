
import torch
import torchcrepe as crepe

#Consts
fmin = 50
fmax = 550

sr = 16000
hop_length = 64

model = 'tiny'

batch_size = None

#decoder = crepe.decode.viterbi
decoder = crepe.decode.argmax

silence_tresh = -60. #dB
periodicity_tresh = .21

crepe.UNVOICED = 0.

def filtered_pitch(signal):
    
    device = signal.device
    
    if signal.ndim == 3:
        squeezed = True
        signal = signal.squeeze(1)
    
    batches = crepe.preprocess(signal,
                               sample_rate = sr,
                               hop_length = hop_length,
                               batch_size = batch_size,
                               device = device,
                               pad = True)
    
    activations_list = []
    pitches_list = []
    periodicities_list = []
    
    for batch in batches:
        
        activations = crepe.infer(batch, model)
        # shape=(batch, 360, time / hop_length)
        activations = activations.reshape(signal.size(0), -1, crepe.PITCH_BINS).transpose(1,2)
        
        activ = torch.clone(activations)
        with torch.no_grad():
            pitches, periodicities = crepe.postprocess(activ,
                                                     fmin, fmax,
                                                     decoder = decoder,
                                                     return_periodicity = True)
        activations_list.append(activations)
        pitches_list.append(pitches)
        periodicities_list.append(periodicities)
        
    
    activations = torch.cat(activations_list, 1)
    pitches = torch.cat(pitches_list, 1)
    periodicities = torch.cat(periodicities_list, 1)
    
    #Filtering pitch
    with torch.no_grad():
        #periodicities = crepe.filter.median(periodicities, 3)
        if signal.shape[0] == 1:
            periodicities = crepe.threshold.Silence(silence_tresh)(periodicities,
                                                                   signal, sr, hop_length)
        
        pitches = crepe.threshold.At(periodicity_tresh)(pitches, periodicities)
    
    if squeezed:
        pitches = pitches.unsqueeze(1)
    pitches = pitches[:,:,:-1].clone()
    return pitches, activations
        
        
def get_shift(pitch_source, pitch_target):
    return crepe.convert.frequency_to_bins(pitch_target) - crepe.convert.frequency_to_bins(pitch_source)

