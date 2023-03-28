import os
import argparse
import random
import pickle
from glob import glob
import tqdm
import soundfile as sf
import numpy as np

import util


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='Root folder of the dataset')
    parser.add_argument('--save_folder', type=str, default='', help='Folder in which to save files. If empty will overwrite original files')
    parser.add_argument('--normalization_db', type=float, help='Db to normalise RMS power of siganls to')
    #parser.add_argument('--ext', type=str, default='.wav', help='Extension to look for')
    
    return parser.parse_args()

def main(opt):
    
    if opt.save_folder:
        os.makedirs(opt.save_folder,exist_ok=True)
    else:
        opt.save_folder = opt.dataset_folder
        
    dirs = [d.name for d in os.scandir(opt.dataset_folder) if d.is_dir() and len(glob(os.path.join(d.path,'**','*.wav'),recursive=True)) > 0]
    dirs.sort()
    
    print("Speakers: ", dirs)
    
    for d in tqdm.tqdm(dirs):
        in_dir = os.path.join(opt.dataset_folder, d)
        out_dir = os.path.join(opt.save_folder, d)
        
        os.makedirs(out_dir, exist_ok=True)
        files = glob(os.path.join(in_dir, '**', '*.wav'),recursive=True)
        
        files.sort()
        for file in files:
            signal, sr = sf.read(file)
            
            if opt.normalization_db is not None:
                signal = util.eq_rms(signal, opt.normalization_db)
                
            if np.isnan(signal).any():
                continue
            out_file = file.replace(in_dir, out_dir)
            sf.write(out_file, signal, sr)
            
    
            
    
if __name__ == '__main__':
    opt = parse_arguments()
    main(opt)