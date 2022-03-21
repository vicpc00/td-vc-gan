import os
import argparse
import glob
import re
import time
import pickle

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def load_dicts(test_dir):
    result_files = ['spkrec_results', 'mosnet_results', 'mcd_results']
    result_dict = {}
    
    for rf in result_files:
        with open(os.path.join(test_dir,rf),'rb') as f:
            tmp_dict = pickle.load(f)
        for k in tmp_dict.keys():
            result_dict[k] = tmp_dict[k]
            
#    with open(os.path.join(test_dir,'mcd_result_conv'),'rb') as f:
#        result_dict['mcd_result_conv'] = pickle.load(f)
#    with open(os.path.join(test_dir,'mcd_result_orig'),'rb') as f:
#        result_dict['mcd_result_orig'] = pickle.load(f)
        
    return result_dict

def main():
    count_self = False
    
    root_dir = 'tests/'
    
    metrics = ['mcd_result_conv', 'emb_dist', 'mos_result_conv']
    
    metric_labels = {'mcd_result_conv':'Mel cepstral distance', 
                     'emb_dist':'Embedding cos similarity', 
                     'mos_result_conv':'Predicted MOS'}
    
    test_list = ['vctk_conv_cont_try39-latent_classifier-freeze_enc',
                 'vctk_reference_adain-vc',
                 'vctk_reference_autovc',
                 'vctk_reference_fragmentvc-10tgts',
                 'vctk_casanova']
    
    test_labels = {'vctk_conv_cont_try15-bot6':'Nosso',
              'vctk_reference_adain-vc':'AdaIN-VC',
              'vctk_reference_autovc':'AutoVC',
              'vctk_reference_fragmentvc-10tgts':'FragmentVC',
              'vctk_casanova':'YourTTS'}
    
    result_dicts = {}
    
    for test in test_list:
        result_dicts[test] = load_dicts(os.path.join(root_dir,test))
        
    formated_results = {}
    for metric in metrics:
        formated_results[metric] = []
        for i,test in enumerate(test_list):
            formated_results[metric].append([])
            for src_spk in result_dicts[test][metric].keys():
                for tgt_spk in result_dicts[test][metric][src_spk].keys():
                    if src_spk == tgt_spk and not count_self: continue
                    formated_results[metric][i] += result_dicts[test][metric][src_spk][tgt_spk]
        
        fig, ax = plt.subplots()
        ax.set_title(metric_labels[metric])
        ax.boxplot(formated_results[metric], labels=test_labels.values(),showmeans=True)

        plt.savefig(os.path.join('.',f'boxplots-{metric}.png'))

if __name__ == '__main__':
    main()
