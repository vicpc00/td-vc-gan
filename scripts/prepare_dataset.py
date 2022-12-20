import os
import argparse
import random
import pickle
from glob import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='Root folder of the dataset')
    parser.add_argument('--save_folder', type=str, default='.', help='Folder in which to save files')
    parser.add_argument('--test_size', type=int, default=3, help='Number of test singnals per class')
    parser.add_argument('--max_tests_ratio', type=float, default=0.1, help='Max ratio between test and test+train set sizes per class')
    parser.add_argument('--test_random', action="store_true", help='Randomize which signals aer taken as test. If false will take the N first alphabetically of each folder')
    parser.add_argument('--out_of_sample_speakers', type=int, default=0, help='Reserve N speakers to not be in training set and create a test_outofsample.txt file')
    parser.add_argument('--ext', type=str, default='.npy', help='Extension to look for')

    opt = parser.parse_args()
    
    os.makedirs(opt.save_folder,exist_ok=True)
    
    dirs = [d.name for d in os.scandir(opt.dataset_folder) if d.is_dir() and len(glob(os.path.join(d.path,'*'+opt.ext),recursive=True)) > 0]
    random.shuffle(dirs)
    dirs, dirs_oos = dirs[opt.out_of_sample_speakers:], dirs[:opt.out_of_sample_speakers]

    dirs.sort()
    print("Speakers: ", dirs)
    if dirs_oos:
        print("Speakers out of sample", dirs_oos)

    spks = dict(zip(dirs, list(range(len(dirs)))))
    spks_oos = dict(zip(dirs_oos, list(range(len(dirs), len(dirs)+len(dirs_oos)))))
    train_set = []
    test_set = []
    oos_set = []

    for d in dirs:
        files = glob(os.path.join(opt.dataset_folder,d,'*'+opt.ext),recursive=True)
        print(d,len(files))
        files.sort()
        if len(files) > 5*opt.test_size:
            if opt.test_random:
                random.shuffle(files)
            test_set += ['|'.join([f,d])+'\n' for f in files[:opt.test_size]]
            train_set += ['|'.join([f,d])+'\n' for f in files[opt.test_size:]]
        else:
            train_set += ['|'.join([f,d])+'\n' for f in files]

    for d in dirs_oos:
        files = glob(os.path.join(opt.dataset_folder,d,'*'+opt.ext),recursive=True)
        print(d,len(files))
        files.sort()
        oos_set += ['|'.join([f,d])+'\n' for f in files]

    with open(os.path.join(opt.save_folder,'train_files'),'w') as f:
        f.writelines(train_set)
    with open(os.path.join(opt.save_folder,'test_files'),'w') as f:
        f.writelines(test_set)
    with open(os.path.join(opt.save_folder,'speakers'),'wb') as f:
        pickle.dump(spks,f)
    if oos_set:
        with open(os.path.join(opt.save_folder,'test_oos_files'),'w') as f:
            f.writelines(oos_set)
        with open(os.path.join(opt.save_folder,'speakers_oos'),'wb') as f:
            pickle.dump(spks_oos,f)
