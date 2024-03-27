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

import data.dataset

class PairsDataset(data.dataset.WaveDataset):
    def __init__(self, pairs_file, labels_file, speaker_file, sample_rate=24000, 
                 max_segment_size = None, return_index = False, 
                 augment_noise = None, silence_threshold=None, normalization_db=None,
                 data_augment = False):

        super().__init__(labels_file, speaker_file, sample_rate, max_segment_size, return_index, augment_noise, silence_threshold, normalization_db, data_augment)

        self.labels_lookup = {filename:label for filename, label in self.dataset}

        with open(pairs_file,'r') as f:
            self.pairs_dataset = list(map(lambda l: l.strip().split('|'),f.readlines()))

        print()

    def __getitem__(self, index):
        conv_name, source_path, target_path = self.pairs_dataset[index]

        source_label = self.spk_dict[self.labels_lookup[source_path]]
        target_label = self.spk_dict[self.labels_lookup[target_path]]

        source_signal = self.load_audio(source_path)
        target_signal = self.load_audio(target_path)

        if self.return_index:
            return torch.FloatTensor(source_signal).unsqueeze(0), source_label, torch.FloatTensor(target_signal).unsqueeze(0), target_label, index

        return torch.FloatTensor(source_signal).unsqueeze(0), source_label, torch.FloatTensor(target_signal).unsqueeze(0), target_label


    def get_convname(self, index):
        conv_name = self.pairs_dataset[index][0]
        #return os.path.basename(file_path)
        return conv_name

    def __len__(self):
        return len(self.pairs_dataset)
    