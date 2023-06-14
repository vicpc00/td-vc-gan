import os
import re
import argparse


from common.build_html_parallel import build_html
from common.test_mcd import test_mcd
from common.test_speaker_rec import test_speaker_rec
from common.test_asr import test_asr

def parse_fn(filename):
    #print(filename)
    phrase_id, src_spk, tgt_spk, sig_type = re.match(r'(\d+)-(\S+)-(\S+)-(orig|conv).wav',os.path.basename(filename)).groups()
    #phrase_id, src_spk, tgt_spk, sig_type = re.match(r'(\d+)-\S+_([MF]\d+)-(?:\S+_([MF]\d+)|X)-(orig|conv).wav',os.path.basename(filename)).groups()
    return phrase_id, src_spk, tgt_spk, sig_type

def name_fn(spk):
    return spk.split('_')[-1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    args = parser.parse_args()

    
    return args


if __name__ == '__main__':
    args = parse_args()
    transc_folder = '/home/victor.costa/data/alcaim-transcriptions/'
    
    test_mcd(os.path.join(args.test_dir,'mcd_results'),os.path.join(args.test_dir,'signals'), parse_fn)
    test_speaker_rec(os.path.join(args.test_dir,'spkrec_results'),os.path.join(args.test_dir,'signals'), parse_fn)
    test_asr(os.path.join(args.test_dir,'asr_results'),os.path.join(args.test_dir,'signals'), transc_folder, parse_fn, name_fn)
    build_html(os.path.join(args.test_dir,'index.html'), args.test_dir, parse_fn, name_fn)

