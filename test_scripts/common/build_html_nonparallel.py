import os
import argparse
import glob
import re
import random

html_header = '''
  <head>
    <title>td-stargan-vc</title>
    <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
      text-align: center;
    }
    #player {
        width: 100px;
    }​
    </style>
  </head>'''
html_body = '''
  <body>
  {}
  </body>
'''

html_table_template = '''
  <table>
    <thead>
    {}
    </thead>
    <tbody>
    {}
    </tbody>
  </table>
'''

html_audio_table_header = '''
    <tr>
      <td rowspan="2" style="text-align:center;">Phrase</td>
      <td rowspan="2" style="text-align:center;">Original<br>Speaker</td>
      <td rowspan="2" style="text-align:center;">Original<br>Signal</td>
      <td colspan="{}" style="text-align:center;">Converted Signals</td>
    </tr>
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_dir')
    parser.add_argument('sig_dir')
    parser.add_argument('save_file')
    parser.add_argument('--num_sigs',type=int)
    args = parser.parse_args()
    return args

def build_table(phrase_list,spks,sig_folder='signals'):
    header = html_audio_table_header.format(len(spks))
    header += '<tr>'
    for tgt_spk in spks:
        header += f'<td>{tgt_spk}</td>\n'
    header += '</tr>\n'
    body = ''
    for src_spk, phrase_id in phrase_list:
        body += '<tr>\n'
        body += f'<td>{phrase_id}</td>\n'
        body += f'<td>{src_spk}</td>\n'
        body += f'<td><audio id="player" controls preload="none"><source src="{sig_folder}/{phrase_id}_{src_spk}-X_orig.wav" /></audio></td>\n'
        for tgt_spk in spks:
            body += f'<td><audio id="player" controls preload="none"><source src="{sig_folder}/{phrase_id}_{src_spk}-{tgt_spk}_conv.wav" /></audio></td>\n'
        body += '</tr>\n'
    return html_table_template.format(header,body)

def build_html(working_dir, out_filename, sig_dir, num_sigs, paired=False):
    
    print(num_sigs)
    
    #Getting speakers
    filelist = glob.glob(os.path.join(working_dir,sig_dir,'*orig.wav'))
    #print(filelist[0])
    spks = set()
    phrase_list = []
    for f in filelist:
        phrase_id, src_spk = re.match(r'(\S+)_(\S+)-X_orig.wav',os.path.basename(f)).groups()
        spks.add(src_spk)
        phrase_list.append((src_spk,phrase_id))
    spks = list(spks)
    spks.sort()
    
    if num_sigs is not None:
        random.seed(1234)
        random.shuffle(phrase_list)
        phrase_list = phrase_list[:num_sigs]
    
    phrase_list.sort()
    
    table = build_table(phrase_list, spks, sig_dir)
    
    html = html_header + html_body.format(table)

    with open(os.path.join(working_dir,out_filename), 'w') as f:
        f.writelines(html)

if __name__ == '__main__':
    args = parse_args()
    
    build_html(args.main_dir, args.save_file, args.sig_dir, args.num_sigs)
