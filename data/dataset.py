import os
import random
import pickle
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import resampy

class WaveDataset(Dataset):

    def __init__(self, dataset_file, speaker_file, mode='wav', sample_rate=24000, max_segment_size = None, return_index = False, augment_noise = None, silence_threshold=None):
        
        with open(speaker_file,'rb') as f:
            self.spk_dict = pickle.load(f)
        with open(dataset_file,'r') as f:
            self.dataset = list(map(lambda l: l.strip().split('|'),f.readlines()))
        self.mode = mode
        self.num_spk = len(self.spk_dict.keys())

        self.sr = sample_rate
        self.return_index = return_index
        self.max_segment_size = max_segment_size
        self.augment = False
        
        self.augment_noise = augment_noise
        self.silence_threshold = silence_threshold

        self.spk_reverse_dict = {}
        for key,val in self.spk_dict.items():
            self.spk_reverse_dict[val] = key
            

    def __getitem__(self, index):
        file_path,label = self.dataset[index]

        if self.mode == 'wav':
            signal,sr = sf.read(file_path)
            if sr != self.sr:
                #print('Warning: sample rate missmatch')
                signal = resampy.resample(signal,sr,self.sr)
        else:
            signal = np.load(file_path).T
        if self.augment:
            G = np.random.uniform(low=0.3, high=1.0)
            signal = signal*G
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
            
        if len(signal[np.abs(signal)>0])==0:
            print(f'All zero signal at signal {self.dataset[index]}, idx {idx}')
        
        if self.augment_noise != None:
            signal = signal + np.random.randn(*signal.shape)*self.augment_noise
        label = self.spk_dict[label]
        if not self.return_index:
            return torch.FloatTensor(signal).unsqueeze(0), label
        else:
            return torch.FloatTensor(signal).unsqueeze(0), label, index

    def __len__(self):
        return len(self.dataset)

    def get_filename(self, index):
        file_path,label = self.dataset[index]
        return os.path.basename(file_path)
    def get_label(self, index):
        _,label = self.dataset[index]
        return label, self.spk_dict[label]

def collate_fn(data):

    signals, labels = zip(*data)
    max_len = max([sig.shape[1] for sig in signals])
    max_len = -256*(-max_len//256)
    signals_pad = [F.pad(sig, (0,max_len - sig.shape[1]), 'constant', value=0) for sig in signals]

    return torch.stack(signals_pad), torch.LongTensor(labels)

    

        
