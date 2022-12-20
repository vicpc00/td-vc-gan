
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_datasets', type=str, nargs='+', help='Source datasets')
    parser.add_argument('target_dataset', type=str, help='Dataset to save')
    parser.add_argument('--root_folder', type=str, default='.', help='Common path to all DSs')
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(os.path.join(args.root_folder, args.target_dataset), exist_ok=True)
    
    file_list = ['train_files', 'test_files']
    
    for fn in file_list: 
        lines = []
        for source_dataset in args.source_datasets:
            with open(os.path.join(args.root_folder, source_dataset, fn), 'r') as f:
                lines += f.readlines()
        with open(os.path.join(args.root_folder, args.target_dataset, fn), 'w') as f:
            f.writelines(lines)
    
    speaker_dict = {}
    offset = 0
    for source_dataset in args.source_datasets:
        with open(os.path.join(args.root_folder, source_dataset, 'speakers'), 'rb') as f:
            speaker_dict_source = pickle.load(f)
        for speaker in speaker_dict_source:
            speaker_dict[speaker] = speaker_dict_source[speaker] + offset
        offset = len(speaker_dict)
    with open(os.path.join(args.root_folder, args.target_dataset, 'speakers'), 'wb') as f:
        pickle.dump(speaker_dict, f)
    
            
if __name__ == '__main__':
    args = parse_args()
    main(args)
    