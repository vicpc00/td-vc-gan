import os
import argparse
import glob
import re
import time
import pickle

import numpy as np
import scipy.stats as st

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args

def get_train_times(model_dir):
    model_files = glob.glob(os.path.join(model_dir, 'step*-G.pt'))
    model_files.sort(key=os.path.getmtime)
    
    model_times = [os.path.getmtime(fn) for fn in model_files]
    
    model_time_diff = [model_times[i+1]-model_times[i] for i in range(len(model_times)-1)]
    
    z = np.abs(st.zscore(model_time_diff))
    z_thresh = 1.5
    model_time_diff = np.array(model_time_diff)
    for ii in range(1,len(model_files)):
        print(model_times[ii]/60/60, model_time_diff[ii-1]/60/60, z[ii-1])
    
    model_time_diff[z>z_thresh] = model_time_diff[z<z_thresh].mean()
    
    model_times = np.cumsum(np.insert(model_time_diff,0,0))
    
    model_epochs = [re.search(r'step(\d+)',model_file).group(1) for model_file in model_files]
        
    return dict(zip(model_epochs, model_times.tolist()))

def get_info(model_dir,save_file):
    info = {}
    train_times = get_train_times(model_dir)
    for e, t in train_times.items():
        print(e,t/60/60)
    info['train_times'] = train_times
    
    info['start_time'] = os.path.getmtime(os.path.join(model_dir, 'step0-G.pt'))
    print(time.strftime('%x %X',time.localtime(info['start_time'])))
    
    with open(os.path.join(model_dir, 'githash')) as f:
        info['git_commit'] = f.readline()
    print(info['git_commit'])
    
    with open(save_file,'wb') as f:
        pickle.dump(info,f)

if __name__ == '__main__':
    args = parse_args()
    
    get_info(args.model_dir,args.save_file)

