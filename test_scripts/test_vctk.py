import os
import re
import argparse


import common.build_html_parallel
import common.test_mcd

def parse_fn(filename):
    phrase_id, src_spk, tgt_spk, sig_type = re.match(r'(\d+)-(\S+)-(\S+)-(orig|conv).wav',os.path.basename(filename)).groups()
    
    return phrase_id, src_spk, tgt_spk, sig_type


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    args = parser.parse_args()

    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    common.test_mcd.test_mcd(os.path.join(args.test_dir,'mcd_results'),os.path.join(args.test_dir,'signals'), parse_fn)
    
    common.build_html_parallel.build_html(os.path.join(args.test_dir,'index-new.html'), args.test_dir, parse_fn)

