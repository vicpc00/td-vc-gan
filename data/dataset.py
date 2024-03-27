import os
import random
import warnings
import pickle
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import resampy

import util
import util.contentvec.audio_corruption as audio_corruption

import parselmouth

import warnings
warnings.filterwarnings("error", category=parselmouth.PraatWarning)
warnings.filterwarnings("error", category=RuntimeWarning)

class WaveDataset(Dataset):


    def __init__(self, dataset_file, speaker_file, sample_rate=24000, 
                 max_segment_size = None, return_index = False, 
                 augment_noise = None, silence_threshold=None, normalization_db=None,
                 data_augment = False, add_new_spks = False, corrupt = False):
        
        with open(speaker_file,'rb') as f:
            self.spk_dict = pickle.load(f)
        with open(dataset_file,'r') as f:
            self.dataset = list(map(lambda l: l.strip().split('|'),f.readlines()))
        self.mode = os.path.splitext(self.dataset[0][0])[1][1:] #extension of first  file without dot
        self.num_spk = len(self.spk_dict.keys())

        self.sr = sample_rate
        self.return_index = return_index
        self.max_segment_size = max_segment_size
        self.min_segment_size = 10*8*2*2*16
        self.segment_multi = 10*8*2*2
        self.augment = False
        self.corrupt = corrupt
        
        self.augment_noise = augment_noise
        self.silence_threshold = silence_threshold
        
        self.normalization_db = normalization_db
        
        self.data_augment = data_augment

        self.spk_reverse_dict = {}
        for key,val in self.spk_dict.items():
            self.spk_reverse_dict[val] = key

        if add_new_spks:
            for file_path,label in self.dataset:
                if label not in self.spk_dict:
                    self.spk_dict[label] = len(self.spk_dict)
                    self.spk_reverse_dict[self.spk_dict[label]] = label
                    #print(self.spk_dict)
            self.num_spk = len(self.spk_dict.keys())
            
        self.corr_succ = 0
        self.corr_warn = 0
        self.corr_err = 0
            
    def corrupt_audio(self, signal, filename):
        try:
            signal_corr = audio_corruption.random_formant_f0(signal, self.sr)
            #print(f'Success - {filename}')
            #self.corr_succ += 1
        except (UserWarning, parselmouth.PraatWarning):
            signal_corr = np.copy(signal)
            #print(f"Praat Warning - {filename}")
            #self.corr_warn += 1
        #except RuntimeWarning:
        #    signal_corr = np.copy(signal)
        #    print(f"NumPy warining - {filename}")
        except (RuntimeError, RuntimeWarning):
            signal_corr = np.copy(signal)
            #print(f"Praat Error - {filename}")
            #self.corr_err += 1
        signal_corr = audio_corruption.random_eq(signal, self.sr)
        signal_corr = util.eq_rms_signals(signal_corr, signal)
        return signal_corr
                

    def __getitem__(self, index):
        file_path,label = self.dataset[index]
        #print(file_path)

        signal = self.load_audio(file_path)

        if self.corrupt:
            signal_corr = self.corrupt_audio(signal, file_path)

        label = self.spk_dict[label]

        if self.return_index:
            return torch.FloatTensor(signal).unsqueeze(0), label, index
        if self.corrupt:
            return torch.FloatTensor(signal).unsqueeze(0), torch.FloatTensor(signal_corr).unsqueeze(0), label
        return torch.FloatTensor(signal).unsqueeze(0), label
    
    def load_audio(self, file_path):

        if self.mode == 'wav' or self.mode == 'flac':
            signal,sr = sf.read(file_path)
            if sr != self.sr:
                #print('Warning: sample rate missmatch')
                signal = resampy.resample(signal,sr,self.sr)
        elif self.mode == 'mp3':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                signal,sr = librosa.load(file_path,sr=self.sr)
        else:
            signal = np.load(file_path).T
        if self.normalization_db:
            signal = util.eq_rms(signal, self.normalization_db)
        if self.data_augment:
            G = np.random.uniform(low=0.3, high=1.0)
            signal = signal*G
            if np.random.randint(2):
                signal = -signal
        #print(signal.shape[0], self.max_segment_size)
        idx = None
        if self.max_segment_size and signal.shape[0] > self.max_segment_size:
            aux_sig = np.zeros(self.max_segment_size)
            #TODO Better zero detection/removal
            while len(aux_sig[np.abs(aux_sig)>0])==0:
                idx = np.random.randint(signal.shape[0] - self.max_segment_size)
                #idx = torch.randint(signal.shape[0] - self.max_segment_size, (1,)).item()
                aux_sig = signal[idx:idx+self.max_segment_size]
            signal = aux_sig
        if signal.shape[0] < self.min_segment_size:
            #print(file_path, signal.shape)
            signal = np.pad(signal, (0, self.min_segment_size - signal.shape[0]), 'constant', constant_values=0)
        if signal.shape[0] % self.segment_multi:
            pad = -self.segment_multi*(-signal.shape[0]//self.segment_multi) - signal.shape[0]
            signal = np.pad(signal, (0, pad), 'constant', constant_values=0)
        rms = np.sqrt((signal**2).mean())
        #print(f'RMS: {rms} / {20*np.log10(rms)} dB')
        if len(signal[np.abs(signal)>0])==0:
            print(f'All zero signal at signal {self.dataset[index]}, idx {idx}')
        
        if self.augment_noise != None:
            signal = signal + np.random.randn(*signal.shape)*self.augment_noise

        return signal
            
        

    def __len__(self):
        return len(self.dataset)

    def get_filename(self, index):
        file_path,label = self.dataset[index]
        #return os.path.basename(file_path)
        return file_path
    def get_label(self, index):
        _,label = self.dataset[index]
        return label, self.spk_dict[label]
    def get_corr_status(self):
        return self.corr_succ, self.corr_warn, self.corr_err
    def reset_corr_status(self):
        self.corr_succ = self.corr_warn = self.corr_err = 0

class SpeakerDataset(WaveDataset):

    #def __init__(self, speaker_id, dataset_file, speaker_file, sample_rate=24000, max_segment_size = None, return_index = False, augment_noise = None, silence_threshold=None):
    def __init__(self, speaker_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #super().__init__(dataset_file, speaker_file, sample_rate, max_segment_size, return_index, augment_noise, silence_threshold)
        
        self.full_dataset = self.dataset
        self.dataset = list(filter(lambda entry: entry[1] == speaker_id, self.full_dataset))

def collate_fn(data):
    data = zip(*data)
    
    ret = []
    for elem in data:
        if type(elem[0]) == torch.Tensor:
            max_len = max([sig.shape[1] for sig in elem])
            signals_pad = [F.pad(sig, (0,max_len - sig.shape[1]), 'constant', value=0) for sig in elem]
            ret.append(torch.stack(signals_pad))
        elif type(elem[0]) == int:
            ret.append(torch.LongTensor(elem))
        else:
            raise Exception("Unrecognized type")
            
    return ret
            
        
def collate_fn_old(data):
    if len(data[0]) == 3:
        signals, labels, idxs = zip(*data)
    else: #len(data[0]) == 2
        signals, labels = zip(*data)
    
    max_len = max([sig.shape[1] for sig in signals])
    #max_len = -1024*(-max_len//1024)
    #max_len
    signals_pad = [F.pad(sig, (0,max_len - sig.shape[1]), 'constant', value=0) for sig in signals]
    

    if len(data[0]) == 3:
        return torch.stack(signals_pad), torch.LongTensor(labels), torch.LongTensor(idxs)
    return torch.stack(signals_pad), torch.LongTensor(labels)
    

        
