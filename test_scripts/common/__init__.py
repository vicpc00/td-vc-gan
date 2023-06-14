import os
import re

def parse_fn(filename):
    phrase_id, src_spk, tgt_spk, sig_type = re.match(r'(\d+)-(\S+)-(\S+)-(orig|conv).wav',os.path.basename(filename)).groups()
    return phrase_id, src_spk, tgt_spk, sig_type

