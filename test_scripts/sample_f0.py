import os
import sys
import argparse
import pickle
import glob
import re
import random

import numpy as np
import pyworld
import librosa

import matplotlib.pyplot as plt

from tqdm import tqdm

ref_f0s = {}

def get_f0(filename, sr=16000):
    
    if filename in ref_f0s.keys():
        return ref_f0s[filename]

    f0_min=50.0
    f0_max = 500.0
    hop_len = 5.0 #ms

    signal,sr = librosa.load(filename, sr=sr)

    signal = signal.astype('double')
    _f0, time_axis = pyworld.dio(signal, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=hop_len)
    f0 = pyworld.stonemask(signal, _f0, time_axis, sr)
    
    f0[f0 == 0] = np.nan

    ref_f0s[filename] = f0
    
    return f0


def mean_ratio(path):
    
    conv_list = glob.glob(os.path.join(path, '*conv.wav'))
    
    mean_of_ratios = []
    ratio_of_means = []
    ratio_of_means_tgt = []
    
    for conv_file in tqdm(conv_list):
        src_spk, phrase_id, tgt_spk = re.match(r'(\S+)_(\d+)_\1-(\S+?)_conv.wav',os.path.basename(conv_file)).groups()
        src_file = os.path.join(os.path.dirname(conv_file),f'{src_spk}_{phrase_id}_{src_spk}-X_orig.wav')
        tgt_file = os.path.join(os.path.dirname(conv_file),f'{tgt_spk}_{phrase_id}_{tgt_spk}-X_orig.wav')
        
        conv_f0 = get_f0(conv_file)
        src_f0 = get_f0(src_file)
        tgt_f0 = get_f0(tgt_file)
        
        mean_of_ratios.append(np.mean(src_f0[np.logical_and(conv_f0==conv_f0, src_f0==src_f0)]/
                                      conv_f0[np.logical_and(conv_f0==conv_f0, src_f0==src_f0)]))
        ratio_of_means.append(np.mean(src_f0[np.logical_and(conv_f0==conv_f0, src_f0==src_f0)])/
                              np.mean(conv_f0[np.logical_and(conv_f0==conv_f0, src_f0==src_f0)]))
        ratio_of_means_tgt.append(np.mean(src_f0[src_f0==src_f0])/
                                  np.mean(tgt_f0[tgt_f0==tgt_f0]))
        
    fig, axs = plt.subplots(1,3, figsize=(12.8,4.8))
    fig.tight_layout()
    axs[0].set_title('Means of ratios - src/conv')
    axs[0].hist(mean_of_ratios, bins=list(np.linspace(-0,3,301)), density=True)
    axs[1].set_title('Ratios of means - src/conv')
    axs[1].hist(ratio_of_means, bins=list(np.linspace(-0,3,301)), density=True)
    axs[2].set_title('Ratios of means - src/tgt')
    axs[2].hist(ratio_of_means_tgt, bins=list(np.linspace(-0,3,301)), density=True)
    
    #plt.savefig(os.path.join('.','f0_ratio_histograms.png'))
    plt.show()
    

def main(path):
    
    conv_list = glob.glob(os.path.join(path, '*conv.wav'))
    
    
    while(True):
        conv_file = random.choice(conv_list)
        print(os.path.basename(conv_file))
        src_spk, phrase_id, tgt_spk = re.match(r'(\S+)_(\d+)_\1-(\S+?)_conv.wav',os.path.basename(conv_file)).groups()
        src_file = os.path.join(os.path.dirname(conv_file),f'{src_spk}_{phrase_id}_{src_spk}-X_orig.wav')
        tgt_file = os.path.join(os.path.dirname(conv_file),f'{tgt_spk}_{phrase_id}_{tgt_spk}-X_orig.wav')
        
        conv_f0 = get_f0(conv_file)
        src_f0 = get_f0(src_file)
        tgt_f0 = get_f0(tgt_file)
        
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (Samples)')
        ax.set_ylabel('F0 (Hz)')
        ax.set_title(f'Speakers: {src_spk}->{tgt_spk}, Phrase: {phrase_id}')
        ax.set_ylim([0,350])
        
        plt.plot(conv_f0, label='Converted')
        plt.plot(src_f0, label='Source')
        plt.plot(tgt_f0, label='Target')
        
        ratio = np.mean(src_f0[np.logical_and(conv_f0==conv_f0, src_f0==src_f0)]/
                        conv_f0[np.logical_and(conv_f0==conv_f0, src_f0==src_f0)])
        textstr = '\n'.join([
            f'Source mean: {np.mean(src_f0[np.isnan(src_f0)==False]):.2f}',
            f'Target mean: {np.mean(tgt_f0[np.isnan(tgt_f0)==False]):.2f}',
            f'Converted mean: {np.mean(conv_f0[np.isnan(conv_f0)==False]):.2f}',
            f'Mean ratio source/conv: {ratio:.2f}'])
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top')
        ax.legend()
        plt.show()
    
    
if __name__ == '__main__':
    main(sys.argv[1])
    #mean_ratio(sys.argv[1])
