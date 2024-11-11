__package__ = "common"

import os
import argparse
import pickle
import glob
import re

import numpy as np
import soundfile as sf
import librosa
import pysptk
import pyworld
from fastdtw import fastdtw
from tqdm import tqdm

from . import parse_fn as default_parse_fn

import warnings


ref_mceps = {}

def eq_rms(signal, target_rms):
    rms = np.sqrt((signal**2).mean())
    gain = 10**(target_rms/20)/rms
    return signal*gain

def world_analyze(signal, sr=16000):
    
    f0_min=50.0
    f0_max = 500.0
    nfft=1024
    hop_len = 5.0 #ms
    
    mcep_dim=24
    mcep_alpha=0.42

    signal = signal.astype('double')
    #f0, time_axis = pyworld.harvest(signal, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=hop_len)
    _f0, time_axis = pyworld.dio(signal, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=hop_len)
    f0 = pyworld.stonemask(signal, _f0, time_axis, sr)
    
    sp = pyworld.cheaptrick(signal, f0, time_axis, sr, fft_size=nfft)  # extract smoothed spectrogram
    #ap = pyworld.d4c(signal, f0, time_axis, sr, fft_size=nfft)         # extract aperiodicity
    
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)
    
    return mcep, f0

    

def mfcc_dist(test_file, ref_file, sr=16000):
    #print(ref_mceps.keys())
    
    if test_file in ref_mceps.keys():
        test_mcep, test_f0 = ref_mceps[test_file]
    else:
        test_signal,sr = librosa.load(test_file, sr=sr)
        
        test_mcep, test_f0 = world_analyze(test_signal, sr)
        test_mcep = test_mcep[test_f0 > 0] #Remove silence
        
    if sum(test_f0 > 0) < 10:
        print("Error: file with no voiced frames:", test_file, ref_file)
        return np.nan, np.nan, np.nan
    
    if ref_file in ref_mceps.keys():
        ref_mcep, ref_f0 = ref_mceps[ref_file]
    else:
        ref_signal,sr = librosa.load(ref_file, sr=sr)
        
        ref_mcep, ref_f0 = world_analyze(ref_signal, sr)
        ref_mcep = ref_mcep[ref_f0 > 0] #Remove silence
        
        ref_mceps[ref_file] = (ref_mcep, ref_f0)

    try:
        (dist, path) = fastdtw(test_mcep, ref_mcep, dist=2)
        if len(path) == 0:
            print("Error: 0 len path with files:", test_file, ref_file)
            return np.nan, np.nan, np.nan
        diff_f0_mean = np.mean(np.log(test_f0[test_f0 > 0])) - np.mean(np.log(ref_f0[ref_f0 > 0]))
        diff_f0_var = np.log(np.var(test_f0[test_f0 > 0])) - np.log(np.var(ref_f0[ref_f0 > 0]))
    except RuntimeWarning:
        print(test_file, ref_file)
        print(np.sum(test_f0 > 0), np.sum(ref_f0 > 0))
        print(len(path))
        return np.nan, np.nan, np.nan
        
    return dist/len(path), diff_f0_mean, diff_f0_var

def f0_ratio(test_file, ref_file, sr=16000):
    
    if test_file in ref_mceps.keys():
        _, test_f0 = ref_mceps[test_file]
    else:
        test_signal,sr = librosa.load(test_file, sr=sr)
        
        test_mcep, test_f0 = world_analyze(test_signal, sr)
        test_mcep = test_mcep[test_f0 > 0] #Remove silence
        
    if sum(test_f0 > 0) < 3:
        return np.nan
    
    if ref_file in ref_mceps.keys():
        _, ref_f0 = ref_mceps[ref_file]
    else:
        ref_signal,sr = librosa.load(ref_file, sr=sr)
        
        ref_mcep, ref_f0 = world_analyze(ref_signal, sr)
        ref_mcep = ref_mcep[ref_f0 > 0] #Remove silence
        
        ref_mceps[ref_file] = (ref_mcep, ref_f0)
    try:
        ratio_f0 = np.mean(ref_f0[ref_f0>0])/np.mean(test_f0[test_f0>0])
    except RuntimeWarning:
        print(test_file, ref_file)
        print(np.sum(test_f0 > 0), np.sum(ref_f0 > 0))
        return np.nan
    
    return ratio_f0

    



def test_mcd(out_filename, test_dir, parse_fn = None):
    warnings.filterwarnings("error", category=DeprecationWarning)
    
    if not parse_fn:
        parse_fn = default_parse_fn
    
    orig_list = glob.glob(os.path.join(test_dir, '*X-orig.wav'))
    orig_list.sort()
    #print(orig_list)
    
    results={'mcd_result_conv':{}, 'mcd_result_orig':{}, 'diff_f0_mean':{}, 'diff_f0_var':{}, 'f0_ratio':{}, 'f0_ratio_orig':{}}
    
    

    for src_file in tqdm(orig_list):
        sig_id, src_spk, _, _ = parse_fn(src_file)

        conv_list =  glob.glob(os.path.join(test_dir, f'{sig_id}-{src_spk}-*-conv.wav'))
        for conv_file in conv_list:
            _, _, tgt_spk, _ = parse_fn(conv_file)
            
            tgt_file = os.path.join(test_dir,f'{sig_id}-{tgt_spk}-X-orig.wav')
            
            mcd_result, diff_f0_mean, diff_f0_var = mfcc_dist(conv_file,tgt_file)
            results['mcd_result_conv'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(mcd_result)
            results['diff_f0_mean'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(diff_f0_mean)
            results['diff_f0_var'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(diff_f0_var)
            
            results['f0_ratio'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(f0_ratio(conv_file,src_file))

    for src_file in tqdm(orig_list):
        sig_id, src_spk, _, _ = parse_fn(src_file)

        for tgt_file in orig_list:
            sig_id_tgt, tgt_spk, _, _ = parse_fn(tgt_file)

            if sig_id != sig_id_tgt:
                continue
            mcd_result, _, _ = mfcc_dist(src_file,tgt_file)
            results['mcd_result_orig'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(mcd_result)
    
            results['f0_ratio_orig'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(f0_ratio(tgt_file,src_file))
    
    with open(out_filename,'wb') as f:
        pickle.dump(results,f)

    #print(dists)
    warnings.resetwarnings()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_mcd(args.save_file, args.test_path)

