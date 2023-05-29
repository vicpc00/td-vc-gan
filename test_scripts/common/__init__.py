import os
import re

def parse_fn(filename):
    sig_id, src_spk, tgt_spk, sig_type = re.match('(\S+)_(\S+)-(\S+)_(orig|conv).wav',os.path.basename(filename)).groups()
    return sig_id, src_spk, tgt_spk, sig_type

