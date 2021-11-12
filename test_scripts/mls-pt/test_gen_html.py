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
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args

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

    </thead>
    <tbody>
    {}
    </tbody>
  </table>
'''

html_table_headers = '''
    <tr>
      <td rowspan="{}" style="text-align:center;">Source<br>Speakers</td>
      <td></td>
      <td colspan="{}" style="text-align:center;">Target Speakers</td>
    </tr>
'''

info_template = '''
      <h2>Training details</h2>
      <p><a href="config.yaml"> Training config file</a><br/>
      Git commit: {}<br/>
      Number of epochs: {}<br/>
      Training time: {:.2f} hours<br/>
      Date: {}</p>
'''

dist_template = '''
      <h3>{}</h3>
      <p>{}</p>
'''

def build_tables(phrase_ids, spks, result_dict):
    tables = ''
    for i, phrase_id in enumerate(phrase_ids):
        table = f'<h3>Phrase {phrase_id}:</h3>\n'
        table += f'<h4>Audio Signals</h4>\n'
        table += build_audio_and_result_table(phrase_id,i, spks, result_dict)
        #table += f'<h4>Audio Signals</h4>\n'
        #table += build_audio_table(phrase_id, spks)
        #table += f'<h4>Mel Cepstral Distances</h4>\n'
        #table += build_result_table(i, spks, result_dict)
        tables += table
    return tables

def build_audio_and_result_table(phrase_id, phrase_idx, spks, result_dict):
    #print(phrase_id, phrase_idx,len(result_dict[spks[0]][spks[0]]))
    table = html_table_headers.format(5*len(spks)+3,len(spks)+1)
    table += '<tr>\n<td></td>\n'
    for tgt_spk in spks:
        table += f'<td>{tgt_spk}</td>\n'
    table += '</tr>\n'
    table += '<tr>\n<td>Originals</td>\n'
    for tgt_spk in spks:
        table += f'<td><audio id="player" controls preload="none"><source src="signals/{tgt_spk}_{phrase_id}_{tgt_spk}-X_orig.wav" /></audio></td>\n'
    table += '</tr>\n'   
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td rowspan="5">{src_spk}</td>\n'
        for tgt_spk in spks:
            table += f'<td><audio id="player" controls preload="none"><source src="signals/{src_spk}_{phrase_id}_{src_spk}-{tgt_spk}_conv.wav" /></audio></td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            correct = result_dict["test_class"][src_spk][tgt_spk][int(phrase_idx)] == tgt_spk
            table += f'<td bgcolor={"green" if correct else "red"}>{result_dict["test_class"][src_spk][tgt_spk][int(phrase_idx)]}</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            table += f'<td>{result_dict["mcd_result_conv"][src_spk][tgt_spk][int(phrase_idx)]:.2f}</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            table += f'<td>{result_dict["emb_dist"][src_spk][tgt_spk][int(phrase_idx)]:.2f}</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            table += f'<td>{result_dict["mos_result_conv"][src_spk][tgt_spk][int(phrase_idx)]:.2f}</td>\n'
        table += '</tr>\n'
    return html_table_template.format(table)

def build_audio_table(phrase_id, spks):
    table = html_table_headers.format(len(spks)+3,len(spks)+1)
    table += '<tr>\n<td></td>\n'
    for tgt_spk in spks:
        table += f'<td>{tgt_spk}</td>\n'
    table += '</tr>\n'
    table += '<tr>\n<td>Originals</td>\n'
    for tgt_spk in spks:
        table += f'<td><audio id="player" controls preload="none"><source src="signals/{tgt_spk}_{phrase_id}_{tgt_spk}-X_orig.wav" /></audio></td>\n'
    table += '</tr>\n'   
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td>{src_spk}</td>\n'
        for tgt_spk in spks:
            table += f'<td><audio id="player" controls preload="none"><source src="signals/{src_spk}_{phrase_id}_{src_spk}-{tgt_spk}_conv.wav" /></audio></td>\n'
        table += '</tr>\n'
    return html_table_template.format(table)

def build_result_table(phrase_id, spks, result_dict):
    table = html_table_headers.format(len(spks)+2,len(spks))
    table += '<tr>\n<td></td>\n'
    for tgt_spk in spks:
        table += f'<td style="width:70px; height:30px">{tgt_spk}</td>\n'
    table += '</tr>\n' 
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td style="width:70px; height:30px">{src_spk}</td>\n'
        for tgt_spk in spks:
            table += f'<td>{result_dict[src_spk][tgt_spk][int(phrase_id)]:.2f}</td>\n'
        table += '</tr>\n'
    return html_table_template.format(table)

def build_sumary_table(result_dict):
    spks = result_dict.keys()
    table = html_table_headers.format(len(spks)+2,len(spks))
    table += '<tr>\n<td></td>\n'
    for tgt_spk in spks:
        table += f'<td style="width:70px; height:30px">{tgt_spk}</td>\n'
    table += '</tr>\n' 
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td style="width:70px; height:30px">{src_spk}</td>\n'
        for tgt_spk in spks:
            table += f'<td>{result_dict[src_spk][tgt_spk]:.2f}</td>\n'
        table += '</tr>\n'
    return html_table_template.format(table)

def dict_stats(result_dict, count_self = True):
    result_list = []
    for src_spk in result_dict.keys():
        for tgt_spk in result_dict[src_spk].keys():
            if src_spk == tgt_spk and not count_self: continue
            result_list += result_dict[src_spk][tgt_spk]
            
    mean = np.mean(result_list)
    ci  = st.t.interval(0.95, len(result_list)-1, loc=mean, scale=st.sem(result_list))
    ci = (ci[1]-ci[0])/2
    median = np.median(result_list)
    maxval = max(result_list)
    minval = min(result_list)
    
    return mean, ci, median, maxval, minval

def dict_stats_per_pair(result_dict):
    pair_mean_dist = dict.fromkeys(result_dict.keys())
    for src_spk in result_dict.keys():
        pair_mean_dist[src_spk] = dict.fromkeys(result_dict[src_spk].keys())
        for tgt_spk in result_dict[src_spk].keys():
            pair_mean_dist[src_spk][tgt_spk] = sum(result_dict[src_spk][tgt_spk])/len(result_dict[src_spk][tgt_spk])

    
    return pair_mean_dist

def dict_correct_rate(result_dict, count_self = True):
    result_list = []
    for src_spk in result_dict.keys():
        for tgt_spk in result_dict[src_spk].keys():
            if src_spk == tgt_spk and not count_self: continue
            result_list += [spk==tgt_spk for spk in result_dict[src_spk][tgt_spk]]
    
    p = sum(result_list)/len(result_list)
    ci = st.binom.interval(0.95, len(result_list), p)
    ci = (ci[1]-ci[0])/2/len(result_list)
    
    return p,ci

def dict_correct_rate_per_pair(result_dict):
    pair_corr_rate = dict.fromkeys(result_dict.keys())
    for src_spk in result_dict.keys():
        pair_corr_rate[src_spk] = dict.fromkeys(result_dict[src_spk].keys())
        for tgt_spk in result_dict[src_spk].keys():
            pair_corr_rate[src_spk][tgt_spk] = sum([spk==tgt_spk for spk in result_dict[src_spk][tgt_spk]])/len(result_dict[src_spk][tgt_spk])

    return pair_corr_rate

def build_result_sumary(result_dict):
    sumary = '<h2>Objective measures sumary</h2>\n'
    sumary += '<b>Speaker recognition correct rate</b>: {:.3f}&pm;{:.03f}<br/>\n'.format(*dict_correct_rate(result_dict['test_class']))
    sumary += '<b>Real valued measure statistics (excluding self transformation):</b><br/>'
    table = '''
      <tr>
        <td style="text-align:center;">Type of measure</td>
        <td style="text-align:center;">Mean</td>
        <td style="text-align:center;">Confidence<br/>Interval</td>
        <td style="text-align:center;">Median</td>
        <td style="text-align:center;">Max</td>
        <td style="text-align:center;">Min</td>
      </tr>\n'''
      
    
    table += '''
      <tr>
        <td style="text-align:center;">Softmax value of correct speaker</td>
        <td style="text-align:center;">{:.2e}</td>
        <td style="text-align:center;">{:.2e}</td>
        <td style="text-align:center;">{:.2e}</td>
        <td style="text-align:center;">{:.2e}</td>
        <td style="text-align:center;">{:.2e}</td>
      </tr>\n'''.format(*dict_stats(result_dict['test_tgt_prob'], False))
      
      
    sumary += html_table_template.format(table)
    
    
    sumary += '<h2>Objective measures per transformation pair</h3>\n'
    
    sumary += '<h3>Speaker recognition correct rate</h2>\n'
    sumary += build_sumary_table(dict_correct_rate_per_pair(result_dict['test_class']))
    
    
    return sumary
      
def load_dicts(test_dir):
    result_files = ['spkrec_results']
    result_dict = {}
    
    for rf in result_files:
        with open(os.path.join(test_dir,rf),'rb') as f:
            tmp_dict = pickle.load(f)
        for k in tmp_dict.keys():
            result_dict[k] = tmp_dict[k]
            
#    with open(os.path.join(test_dir,'mcd_result_conv'),'rb') as f:
#        result_dict['mcd_result_conv'] = pickle.load(f)
#    with open(os.path.join(test_dir,'mcd_result_orig'),'rb') as f:
#        result_dict['mcd_result_orig'] = pickle.load(f)
        
    return result_dict

def build_html(out_filename, test_dir):
    
    #Getting speakers
    filelist = glob.glob(os.path.join(test_dir,'signals/*orig.wav'))
    print(filelist[0])
    phrase_ids = set()
    spks = set()
    for f in filelist:
        #fn, src_spk, tgt_spk, sig_type = re.match('(\S+)_(\S+?)-(\S+?|X)_(conv|orig).wav',os.path.basename(f))
        src_spk, phrase_id = re.match(r'(\d+)_(\d+_\d+)_\1-X_orig.wav',os.path.basename(f)).groups()
        phrase_ids.add(phrase_id)
        spks.add(src_spk)
    spks = list(spks)
    spks.sort()
    phrase_ids = list(phrase_ids)
    phrase_ids.sort()
    print(phrase_ids,spks)
    
    result_dict = load_dicts(test_dir)
        
    
    """
    with open(os.path.join(test_dir,'mcd_result_orig'),'rb') as f:
        result_dict_orig = pickle.load(f)
    mean_dists_orig = dict.fromkeys(result_dict.keys())
    for src_spk in spks:
        mean_dists_orig[src_spk] = dict.fromkeys(result_dict[src_spk].keys())
        for tgt_spk in spks:
            mean_dists_orig[src_spk][tgt_spk] = [sum(result_dict_orig[src_spk][tgt_spk])/len(result_dict_orig[src_spk][tgt_spk])]
    """
    """
    with open(os.path.join(test_dir,'info'),'rb') as f:
        info_dict = pickle.load(f)
    with open(os.path.join(test_dir,'epoch'),'r') as f:
        epoch = f.read().rstrip()
    
    info = info_template.format(info_dict['git_commit'], epoch, info_dict['train_times'][epoch]/(60*60), time.strftime('%x %X',time.localtime(info_dict['start_time'])))
    """
    info = ''
    '''
    info += dist_template.format('Mean Mel Cepstral Distances between coverted signals', mean_dist)
    info += build_result_table(0, spks, mean_dists)
    info += '<h3>Mean Mel Cepstral Distances between non coverted signals</h3>'
    info += build_result_table(0, spks, mean_dists_orig)
    '''
    
    info += build_result_sumary(result_dict)
    
    tables = ''
    #tables = build_tables(phrase_ids, spks, result_dict)
    
    html = html_header + html_body.format(info+tables)
    
    
    with open(out_filename, 'w') as f:
        f.writelines(html)
    
    return html
  
if __name__ == '__main__':
    args = parse_args()
    
    build_html(args.save_file, args.test_dir)
