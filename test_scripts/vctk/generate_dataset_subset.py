import os
import argparse
import random

fixed_spks = ['p243','p283','p297','p300','p306','p311','p334']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('--num_speakers', default=True, type=int)
    parser.add_argument('--num_phrases', required=True, type=int)
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    #random.seed(1234)
    if args.seed:
        random.seed(args.seed)
    
    dataset = {}
    with open(args.dataset_file,'r') as f:
        for spk, fn in map(lambda l: l.strip().split('|')[::-1],f.readlines()):
            if not spk in dataset.keys():
                dataset[spk] = [fn]
            else:
                dataset[spk].append(fn)
    for k in dataset.keys():
        dataset[spk].sort()
    valid_spks = [spk for spk in dataset.keys() if os.path.basename(dataset[spk][args.num_phrases-1]) == '{}_{:03d}.wav'.format(spk,args.num_phrases)]
    for spk in fixed_spks:
        if spk not in valid_spks: print('Warning: fixed spk {} not valid'.format(spk))
    valid_spks = [spk for spk in valid_spks if spk not in fixed_spks]
    
    random.shuffle(valid_spks)
    
    used_spks = valid_spks[:args.num_speakers-len(fixed_spks)]+fixed_spks

    used_spks.sort()
    
    print(used_spks)

    
    with open(args.out_file,'w') as f:
        f.writelines(['|'.join((fn,spk))+'\n' for spk in used_spks for fn in dataset[spk][:args.num_phrases]])


    with open('/home/victor.costa/code/datasets/vctk/speaker-info.txt','r') as f:
        for line in f.readlines():
            if 'p'+line.split()[0] in used_spks: print(line,end='') 

if __name__ == '__main__':
    main()