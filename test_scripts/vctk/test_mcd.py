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
        test_mcep = ref_mceps[test_file]
    else:
        test_signal,sr = librosa.load(test_file, sr=sr)
        
        test_mcep, test_f0 = world_analyze(test_signal, sr)
        test_mcep = test_mcep[test_f0 > 0] #Remove silence
    
    if ref_file in ref_mceps.keys():
        ref_mcep = ref_mceps[ref_file]
    else:
        ref_signal,sr = librosa.load(ref_file, sr=sr)
        
        ref_mcep, ref_f0 = world_analyze(ref_signal, sr)
        ref_mcep = ref_mcep[ref_f0 > 0] #Remove silence
        
        ref_mceps[ref_file] = ref_mcep
    
    (dist, path) = fastdtw(test_mcep, ref_mcep, dist=2)
    
    return dist/len(path)


def mfcc_dist_old(test_file, ref_file, sr=16000, target_rms=-25):
    
    test_signal,sr = librosa.load(test_file, sr=sr)
    ref_signal,sr = librosa.load(ref_file, sr=sr)
    
    #test_signal = eq_rms(test_signal,target_rms)
    #ref_signal = eq_rms(ref_signal,target_rms)
    
    rms_test = np.sqrt((test_signal**2).mean())
    rms_ref = np.sqrt((ref_signal**2).mean())
    
    test_signal = test_signal*(rms_test+rms_ref)/(2*rms_test)
    ref_signal = ref_signal*(rms_test+rms_ref)/(2*rms_ref)
    
    test_mfcc = librosa.feature.mfcc(y=test_signal, sr=sr).T
    ref_mfcc = librosa.feature.mfcc(y=ref_signal, sr=sr).T
    
    (dist, path) = fastdtw(test_mfcc, ref_mfcc, dist=2)
    
    return dist/len(path)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    orig_list = glob.glob(os.path.join(args.test_path, '*X_orig.wav'))
    orig_list.sort()
    #print(orig_list)
    
    results={'mcd_result_conv':{}, 'mcd_result_orig':{}}
    
    

    for src_file in tqdm(orig_list):
        filename, src_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(src_file)).groups()

        #print(filename,src_spk)
        conv_list =  glob.glob(os.path.join(args.test_path, f'{filename}_{src_spk}-*_conv.wav'))
        #print(conv_list)
        for conv_file in conv_list:
            #print(f'{filename}_{src_spk}-(\S+?)_conv.wav',conv_file)
            tgt_spk = re.match(f'{filename}_{src_spk}-(\S+?)_conv.wav',os.path.basename(conv_file)).group(1)
            
            tgt_file = os.path.join(args.test_path,f'{re.sub(src_spk,tgt_spk,filename)}_{tgt_spk}-X_orig.wav')
            #print(src_file, tgt_file, conv_file)
            
            results['mcd_result_conv'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(mfcc_dist(conv_file,tgt_file))

    for src_file in tqdm(orig_list):
        filename_src, src_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(src_file)).groups()

        #print(filename_src,src_spk)
        for tgt_file in orig_list:
            filename_tgt, tgt_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(tgt_file)).groups()

            if filename_src.split('_')[1] != filename_tgt.split('_')[1]:
                continue
            results['mcd_result_orig'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(mfcc_dist(src_file,tgt_file))
    
    with open(args.save_file,'wb') as f:
        pickle.dump(results,f)

    #print(dists)

if __name__ == '__main__':
    main()
