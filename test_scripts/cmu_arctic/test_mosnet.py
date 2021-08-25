import os
import argparse
import pickle
import glob
import re

import numpy as np

#TODO: Use speechmetrics lib for MOSnet
"""
import speechmetrics
mosnet = speechmetrics.load("mosnet", None)
score = metrics(file_path)["mosnet"][0][0]
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    with open(args.test_path,'r') as f:
        mos_list = f.readlines()
    results = {'mos_result_conv':{}, 'mos_result_orig':{},}
    for mos_eval in mos_list[:-1]:
        filepath, mos = mos_eval.strip().split()
        filename = os.path.basename(filepath)
        phrase_id, src_spk, tgt_spk, sig_type = re.match('(\S+)_cmu_us_(\S+)_arctic-(?:cmu_us_(\S+)_arctic|X)_(conv|orig).wav',filename).groups()
        
        if sig_type == 'conv':
            results['mos_result_conv'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(float(mos))
        elif sig_type == 'orig':
            results['mos_result_orig'].setdefault(src_spk,[]).append(float(mos))
            
    with open(args.save_file,'wb') as f:
        pickle.dump(results,f)
        

if __name__ == '__main__':
    main()

