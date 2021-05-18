import os
import argparse
import pickle
import glob
import re

import numpy as np
import soundfile as sf
import librosa
from fastdtw import fastdtw
from tqdm import tqdm

def eq_rms(signal, target_rms):
    rms = np.sqrt((signal**2).mean())
    gain = 10**(target_rms/20)/rms
    return signal*gain

def mfcc_dist(test_file, ref_file, sr=16000, target_rms=-25):
    
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
    
    dists={}

    for src_file in tqdm(orig_list):
        filename, src_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(src_file)).groups()
        
        if src_spk not in dists.keys():
            dists[src_spk] = {}
        #print(filename,src_spk)
        conv_list =  glob.glob(os.path.join(args.test_path, f'{filename}_{src_spk}-*_conv.wav'))
        #print(conv_list)
        for conv_file in conv_list:
            #print(f'{filename}_{src_spk}-(\S+?)_conv.wav',conv_file)
            tgt_spk = re.match(f'{filename}_{src_spk}-(\S+?)_conv.wav',os.path.basename(conv_file)).group(1)
            
            if tgt_spk not in dists[src_spk].keys():
                dists[src_spk][tgt_spk] = []
            
            tgt_file = os.path.join(args.test_path,f'{re.sub(src_spk,tgt_spk,filename)}_{tgt_spk}-X_orig.wav')
            #print(src_file, tgt_file, conv_file)
            
            dists[src_spk][tgt_spk].append(mfcc_dist(conv_file,tgt_file))

    dists_orig={}      
    for src_file in tqdm(orig_list):
        filename_src, src_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(src_file)).groups()
        if src_spk not in dists_orig.keys():
            dists_orig[src_spk] = {}
        #print(filename_src,src_spk)
        for tgt_file in orig_list:
            filename_tgt, tgt_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(tgt_file)).groups()
            if tgt_spk not in dists_orig[src_spk].keys():
                dists_orig[src_spk][tgt_spk] = []
            if filename_src.split('_')[1] != filename_tgt.split('_')[1]:
                continue
            dists_orig[src_spk][tgt_spk].append(mfcc_dist(src_file,tgt_file))
    
    with open(args.save_file,'wb') as f:
        pickle.dump(dists,f)
    with open(args.save_file+'_orig','wb') as f:
        pickle.dump(dists_orig,f)
    #print(dists)

if __name__ == '__main__':
    main()
