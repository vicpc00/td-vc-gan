import os
import argparse
import pickle
import glob
import re
import csv


transcript_dict_file = '/home/victor.costa/code/datasets/vctk/vctk_transcripts_dict'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args

def prepare_asr_test():
    args = parse_args()
    
    with open(transcript_dict_file,'rb') as f:
        transcript_dict = pickle.load(f)
    
    #sb_hparams = speechbrain_init(args.speechbrain_hparam)

    orig_list = glob.glob(os.path.join(args.test_path, '*X_orig.wav'))
    orig_list.sort()
    #print(orig_list)
    
    conv_csv = []
    orig_csv = []
    
    for src_file in orig_list:
        filename, src_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(src_file)).groups()
        orig_csv.append([os.path.abspath(src_file),os.path.getsize(src_file),transcript_dict[filename].lower().translate(str.maketrans('', '', ',.'))])
        
        #print(filename,transcript_dict[filename])
        conv_list =  glob.glob(os.path.join(args.test_path, f'{filename}_{src_spk}-*_conv.wav'))
        #print(conv_list)
        for conv_file in conv_list:
            #print(f'{filename}_{src_spk}-(\S+?)_conv.wav',conv_file)
            #tgt_spk = re.match(f'{filename}_{src_spk}-(\S+?)_conv.wav',os.path.basename(conv_file)).group(1)
            
            transc = transcript_dict[filename].translate(str.maketrans('', '', ',.'))
            transc = transc.lower()
            
            conv_csv.append([os.path.abspath(conv_file),os.path.getsize(conv_file),transc])
            
    with open(args.save_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        
        csvwriter.writerow(['wav_filename','wav_filesize','transcript'])
        for row in conv_csv:
            csvwriter.writerow(row)
            
if __name__ == '__main__':
    prepare_asr_test()