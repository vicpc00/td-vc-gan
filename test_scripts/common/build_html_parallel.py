import os
import argparse
import glob
import re
import time
import pickle

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--save_file')
    args = parser.parse_args()
    if args.save_file is None:
        args.save_file = os.path.join(args.test_dir,'index.html')
    
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

fig_template = '''
       <figure>
         <img src="{}" style="scale:100%">
         <figcaption>{}</figcaption>
       </figure> 
'''

def build_tables(phrase_ids, spks, result_dict, name_fn):
    tables = ''
    for i, phrase_id in enumerate(phrase_ids):
        table = f'<h3>Phrase {phrase_id}:</h3>\n'
        table += f'<h4>Audio Signals</h4>\n'
        table += build_audio_and_result_table(phrase_id,i, spks, result_dict, name_fn)
        #table += f'<h4>Audio Signals</h4>\n'
        #table += build_audio_table(phrase_id, spks)
        #table += f'<h4>Mel Cepstral Distances</h4>\n'
        #table += build_result_table(i, spks, result_dict)
        tables += table
    return tables

def build_audio_and_result_table(phrase_id, phrase_idx, spks, result_dict, name_fn):
    #print(phrase_id, phrase_idx,len(result_dict[spks[0]][spks[0]]))
    table = html_table_headers.format(5*len(spks)+3,len(spks)+1)
    table += '<tr>\n<td></td>\n'
    for tgt_spk in spks:
        table += f'<td>{name_fn(tgt_spk)}</td>\n'
    table += '</tr>\n'
    table += '<tr>\n<td>Originals</td>\n'
    for tgt_spk in spks:
        table += f'<td><audio id="player" controls preload="none"><source src="signals/{phrase_id}-{tgt_spk}-X-orig.wav" /></audio></td>\n'
    table += '</tr>\n'   
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td rowspan="5">{name_fn(src_spk)}</td>\n'
        for tgt_spk in spks:
            table += f'<td><audio id="player" controls preload="none"><source src="signals/{phrase_id}-{src_spk}-{tgt_spk}-conv.wav" /></audio></td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            if 'test_class' in result_dict:
                correct = result_dict["test_class"][src_spk][tgt_spk][int(phrase_idx)] == tgt_spk
                table += f'<td bgcolor={"green" if correct else "red"}>{name_fn(result_dict["test_class"][src_spk][tgt_spk][int(phrase_idx)])}</td>\n'
            else:
                table += '<td>&mdash;</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            if 'mcd_result_conv' in result_dict:
                table += f'<td>{result_dict["mcd_result_conv"][src_spk][tgt_spk][int(phrase_idx)]:.2f}</td>\n'
            else:
                table += '<td>&mdash;</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            if 'emb_dist' in result_dict:
                table += f'<td>{result_dict["emb_dist"][src_spk][tgt_spk][int(phrase_idx)]:.2f}</td>\n'
            else:
                table += '<td>&mdash;</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        for tgt_spk in spks:
            if 'mos_result_conv' in result_dict:
                table += f'<td>{result_dict["mos_result_conv"][src_spk][tgt_spk][int(phrase_idx)]:.2f}</td>\n'
            else:
                table += '<td>&mdash;</td>\n'
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
        table += f'<td><audio id="player" controls preload="none"><source src="signals/{phrase_id}-{tgt_spk}-X-orig.wav" /></audio></td>\n'
    table += '</tr>\n'   
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td>{src_spk}</td>\n'
        for tgt_spk in spks:
            table += f'<td><audio id="player" controls preload="none"><source src="signals/{phrase_id}-{src_spk}-{tgt_spk}-conv.wav" /></audio></td>\n'
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

def build_sumary_table(result_dict, spks, name_fn):
    #spks = result_dict.keys()
    table = html_table_headers.format(len(spks)+2,len(spks))
    table += '<tr>\n<td></td>\n'
    for tgt_spk in spks:
        table += f'<td style="width:70px; height:30px">{name_fn(tgt_spk)}</td>\n'
    table += '</tr>\n' 
        
    for src_spk in spks:
        table += '<tr>\n'
        table += f'<td style="width:70px; height:30px">{name_fn(src_spk)}</td>\n'
        for tgt_spk in spks:
            table += f'<td>{result_dict[src_spk][tgt_spk]:.2f}</td>\n'
        table += '</tr>\n'
    return html_table_template.format(table)

def dict_stats(result_dict, count_self = True, transf = lambda x: x):
    result_list = []
    for src_spk in result_dict.keys():
        for tgt_spk in result_dict[src_spk].keys():
            if src_spk == tgt_spk and not count_self: continue
            result_list += filter(np.isfinite, transf(result_dict[src_spk][tgt_spk]))
            
    mean = np.mean(result_list)
    #ci  = st.t.interval(0.95, len(result_list)-1, loc=mean, scale=st.sem(result_list))
    ci  = st.norm.interval(0.95, loc=mean, scale=st.sem(result_list))
    ci = (ci[1]-ci[0])/2
    std = np.std(result_list)
    median = np.median(result_list)
    maxval = max(result_list)
    minval = min(result_list)
    
    return mean, ci, std, median, maxval, minval

def dict_stats_per_pair(result_dict):
    pair_mean_dist = dict.fromkeys(result_dict.keys())
    for src_spk in result_dict.keys():
        pair_mean_dist[src_spk] = dict.fromkeys(result_dict[src_spk].keys())
        for tgt_spk in result_dict[src_spk].keys():
            filtered = list(filter(np.isfinite, result_dict[src_spk][tgt_spk]))
            pair_mean_dist[src_spk][tgt_spk] = sum(filtered)/len(filtered)

    
    return pair_mean_dist

def dict_correct_rate(result_dict, count_self = True):
    result_list = []
    for src_spk in result_dict.keys():
        if type(result_dict[src_spk]) == dict:
            for tgt_spk in result_dict[src_spk].keys():
                if src_spk == tgt_spk and not count_self: continue
                result_list += [spk==tgt_spk for spk in result_dict[src_spk][tgt_spk]]
        elif type(result_dict[src_spk]) == list:
            result_list += [spk==src_spk for spk in result_dict[src_spk]]
    
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

def build_result_sumary(result_dict, spks, name_fn):
    sumary = '<h2>Objective measures sumary</h2>\n'
    if 'test_class' in result_dict:
        sumary += '<b>Speaker recognition correct rate</b>: {:.3f}&pm;{:.03f}<br/>\n'.format(*dict_correct_rate(result_dict['test_class']))
    if 'asr_results_wer' in result_dict:
        sumary += '<b>Automatic speech recognition word error rate</b>: {:.3f}<br/>\n'.format(result_dict['asr_results_wer'])
        sumary += '<b>Automatic speech recognition character error rate</b>: {:.3f}<br/>\n'.format(result_dict['asr_results_cer'])
    sumary += '<b>Real valued measure statistics (excluding self transformation):</b><br/>'
    table = '''
      <tr>
        <td style="text-align:center;">Type of measure</td>
        <td style="text-align:center;">Mean</td>
        <td style="text-align:center;">Mean<br/>Confidence<br/>Interval</td>
        <td style="text-align:center;">Standard<br/>Deviation</td>
        <td style="text-align:center;">Median</td>
        <td style="text-align:center;">Max</td>
        <td style="text-align:center;">Min</td>
      </tr>\n'''

    if 'mcd_result_conv' in result_dict:      
        table += '''
          <tr>
            <td style="text-align:center;">Mel cepstral distance</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['mcd_result_conv'], False))

    if 'diff_f0_mean' in result_dict:
        table += '''
          <tr>
            <td style="text-align:center;">Diff of log mean F0</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['diff_f0_mean'], False))
          
    if 'diff_f0_mean' in result_dict:
        table += '''
          <tr>
            <td style="text-align:center;">Abs of diff of log mean F0</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['diff_f0_mean'], False, transf = np.abs))

    if 'diff_f0_var' in result_dict:      
        table += '''
          <tr>
            <td style="text-align:center;">Diff of log var F0</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['diff_f0_var'], False))

    if 'emb_dist' in result_dict:      
        table += '''
          <tr>
            <td style="text-align:center;">Embedding cos similarity</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['emb_dist'], False))

    if 'mos_result_conv' in result_dict:      
        table += '''
          <tr>
            <td style="text-align:center;">Predicted MOS</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['mos_result_conv'], False))
      
    sumary += html_table_template.format(table)
    
    sumary += '<h3>Mesures for reference signals</h3>\n'
    if 'ref_class' in result_dict:
        sumary += '<b>Speaker recognition correct rate</b>: {:.2f}&pm;{:.03f}<br/>\n'.format(*dict_correct_rate(result_dict['ref_class']))
    sumary += '<b>Real valued measure statistics (excluding self transformation):</b><br/>'
    table = '''
      <tr>
        <td style="text-align:center;">Type of measure</td>
        <td style="text-align:center;">Mean</td>
        <td style="text-align:center;">Mean<br/>Confidence<br/>Interval</td>
        <td style="text-align:center;">Standard<br/>Deviation</td>
        <td style="text-align:center;">Median</td>
        <td style="text-align:center;">Max</td>
        <td style="text-align:center;">Min</td>
      </tr>\n'''
    
    if 'mcd_result_orig' in result_dict:
        table += '''
          <tr>
            <td style="text-align:center;">Mel cepstral distance (source speaker)</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
            <td style="text-align:center;">{:.3f}</td>
          </tr>\n'''.format(*dict_stats(result_dict['mcd_result_orig'], False))
      
      
    sumary += html_table_template.format(table)
    
    
    sumary += '<h2>Objective measures per transformation pair</h3>\n'
    
    if 'test_class' in result_dict:
        sumary += '<h3>Speaker recognition correct rate</h3>\n'
        sumary += build_sumary_table(dict_correct_rate_per_pair(result_dict['test_class']), spks, name_fn)
    if 'mcd_result_conv' in result_dict:
        sumary += '<h3>Mel cepstral distance</h3>\n'
        sumary += build_sumary_table(dict_stats_per_pair(result_dict['mcd_result_conv']), spks, name_fn)
    if 'diff_f0_mean' in result_dict:
        sumary += '<h3>Diff log F0</h3>\n'
        sumary += build_sumary_table(dict_stats_per_pair(result_dict['diff_f0_mean']), spks, name_fn)
    if 'emb_dist' in result_dict:
        sumary += '<h3>Embedding similarity</h3>\n'
        sumary += build_sumary_table(dict_stats_per_pair(result_dict['emb_dist']), spks, name_fn)
    if 'mos_result_conv' in result_dict:
        sumary += '<h3>Predicted MOS</h3>\n'
        sumary += build_sumary_table(dict_stats_per_pair(result_dict['mos_result_conv']), spks, name_fn)
    if 'asr_results_wer_pair' in result_dict:
        sumary += '<h3>Automatic speech recognition word error rate</h3>\n'
        sumary += build_sumary_table(result_dict['asr_results_wer_pair'], spks, name_fn)
        sumary += '<h3>Automatic speech recognition character error rate</h3>\n'
        sumary += build_sumary_table(result_dict['asr_results_cer_pair'], spks, name_fn)
    
    
    return sumary

def build_plots(result_dict, spks, test_dir):
    
    gen_scatter(result_dict, spks, test_dir)
    gen_boxplots(result_dict, spks, test_dir)
    gen_hists(result_dict, spks, test_dir)
    gen_hist_f0_ratio(result_dict, spks, test_dir)
    
    plots = '<h2>Objective measures plots</h2>\n'
    if os.path.exists(os.path.join(test_dir, 'histograms.png')):
        plots += '<h4>Histograms of measures</h4>\n'
        plots += fig_template.format('histograms.png', '')
    if os.path.exists(os.path.join(test_dir, 'boxplots.png')):
        plots += '<h4>Boxplot of measures</h4>\n'
        plots += fig_template.format('boxplots.png', '')   
    if os.path.exists(os.path.join(test_dir, 'embsim_mos_scatter.png')):
        plots += '<h4>Predicted MOS vs Embedding cos similarity scatter plot</h4>\n'
        plots += fig_template.format('embsim_mos_scatter.png', '')
    return plots
      
def gen_scatter(result_dict, spks, test_dir):
    
    if 'emb_dist' not in result_dict or 'mos_result_conv' not in result_dict:
        return
    
    fig, ax = plt.subplots()
    ax.set_title('Embedding Similarity vs. MOSNet Result')
    ax.set(xlabel='Embedding Similarity', ylabel='MOSNet Result')
    ax.set_ylim([1,5])
    ax.set_xlim([0,1])
    
    for src_spk in spks:
        for tgt_spk in spks:
            plt.scatter(result_dict['emb_dist'][src_spk][tgt_spk], result_dict['mos_result_conv'][src_spk][tgt_spk]
                       , c='blue')
    plt.savefig(os.path.join(test_dir,'embsim_mos_scatter.png'))

def gen_boxplots(result_dict, spks, test_dir):
    count_self = False
    flattened = {}
    
    for res in ['mcd_result_conv', 'emb_dist', 'mos_result_conv']:
        if res not in result_dict:
            continue
        flattened[res] = []
        for src_spk in spks:
            for tgt_spk in spks:
                if src_spk == tgt_spk and not count_self: continue
                flattened[res] += filter(np.isfinite, result_dict[res][src_spk][tgt_spk])
                
    if not flattened:
        return
    
    titles = {'mcd_result_conv': 'Mel cepstral distance',
              'emb_dist': 'Embedding cos similarity',
              'mos_result_conv': 'Predicted MOS'}
    ylims = {'mcd_result_conv': (0.5, 4),
              'emb_dist': (0,1),
              'mos_result_conv': (1,5)}
    
    
    fig, axs = plt.subplots(1,len(flattened), squeeze=False)
    axs = axs.squeeze(0)
    fig.tight_layout()
    
    for idx, res in enumerate(flattened):
        axs[idx].set_title(titles[res])
        axs[idx].set_ylim(*ylims[res])
        axs[idx].boxplot(flattened[res], labels=[''])
    
    plt.savefig(os.path.join(test_dir,'boxplots.png'))
    
def gen_hists(result_dict, spks, test_dir):
    count_self = False
    flattened = {}
    
    for res in ['mcd_result_conv', 'emb_dist', 'mos_result_conv']:
        if res not in result_dict:
            continue
        flattened[res] = []
        for src_spk in spks:
            for tgt_spk in spks:
                if src_spk == tgt_spk and not count_self: continue
                flattened[res] += result_dict[res][src_spk][tgt_spk]
                
    if not flattened:
        return
    
    n = len(flattened)
    fig, axs = plt.subplots(1,n, figsize=(4*n,4), squeeze=False)
    axs = axs.squeeze(0)
    fig.tight_layout()
    idx = 0
    
    titles = {'mcd_result_conv': 'Mel cepstral distance',
              'emb_dist': 'Embedding cos similarity',
              'mos_result_conv': 'Predicted MOS'}
    bins = {'mcd_result_conv': list(np.linspace(0,4,101)),
              'emb_dist': list(np.linspace(0,1,101)),
              'mos_result_conv': list(np.linspace(1,5,101))}
    
    for idx, res in enumerate(flattened):
        axs[idx].set_title(titles[res])
        axs[idx].hist(flattened[res], bins=bins[res], density=True)

    plt.savefig(os.path.join(test_dir,'histograms.png'))
    
def gen_hist_f0_ratio(result_dict, spks, test_dir):
    count_self = False
    flattened = {}
    
    if 'f0_ratio' not in result_dict:
        return
    
    for res in ['f0_ratio', 'f0_ratio_orig', 'diff_f0_mean']:
        flattened[res] = []
        for src_spk in spks:
            for tgt_spk in spks:
                if src_spk == tgt_spk and not count_self: continue
                flattened[res] += result_dict[res][src_spk][tgt_spk]
                
    fig, axs = plt.subplots(1,3, figsize=(12.8,4.8))
    fig.tight_layout()
    axs[0].set_title('Ratio of mean F0 - src/conv')
    axs[0].hist(flattened['f0_ratio'], bins=list(np.linspace(0,3,301)), density=True)
    axs[1].set_title('Ratio of mean F0 - src/tgt')
    axs[1].hist(flattened['f0_ratio_orig'], bins=list(np.linspace(0,3,301)), density=True)
    axs[2].set_title('Difference between mean of logF0 - conv-tgt')
    axs[2].hist(flattened['diff_f0_mean'], bins=list(np.linspace(-1.5,1.5,301)), density=True)
    

    
    plt.savefig(os.path.join(test_dir,'histograms_f0_ratio.png'))

def load_dicts(test_dir):
    result_files = ['spkrec_results', 'mosnet_results', 'mcd_results', 'asr_results']
    result_dict = {}
    
    for rf in result_files:
        rf = os.path.join(test_dir, rf)
        if os.path.exists(rf):
            with open(rf,'rb') as f:
                tmp_dict = pickle.load(f)
            for k in tmp_dict.keys():
                result_dict[k] = tmp_dict[k]
        
    return result_dict

def build_html(out_filename, test_dir, parse_fn, name_fn = None):
    if not name_fn:
        name_fn = lambda x: x
    
    #Getting speakers
    filelist = glob.glob(os.path.join(test_dir,'signals/*orig.wav'))
    phrase_ids = set()
    spks = set()
    for f in filelist:
        phrase_id, src_spk, _, _ = parse_fn(f)
        #fn, src_spk, tgt_spk, sig_type = re.match('(\S+)_(\S+?)-(\S+?|X)_(conv|orig).wav',os.path.basename(f))
        #src_spk, phrase_id = re.match(r'(\S+)_(\d+)_\1-X_orig.wav',os.path.basename(f)).groups()
        phrase_ids.add(phrase_id)
        spks.add(src_spk)
    spks = list(spks)
    spks.sort(key=name_fn)
    phrase_ids = list(phrase_ids)
    phrase_ids.sort()
    #print(phrase_ids,spks)
    
    result_dict = load_dicts(test_dir)
    print(result_dict.keys())
        
    
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
    
    info += build_result_sumary(result_dict, spks, name_fn)
    
    tables = build_tables(phrase_ids, spks, result_dict, name_fn)
    
    plots = build_plots(result_dict, spks, test_dir)
    
    html = html_header + html_body.format(info+plots+tables)
        
    with open(out_filename, 'w') as f:
        f.writelines(html)
    
    return html
  
if __name__ == '__main__':
    args = parse_args()
    
    build_html(args.save_file, args.test_dir)
