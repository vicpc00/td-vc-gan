import os
import argparse
from pathlib import Path
import re

import soundfile as sf

import torch
import numpy as np
from tqdm import tqdm

from model.generator import Generator
import data.dataset as dataset

from util.hparams import HParam

def label2onehot(labels, n_classes):
    #labels: (batch_size,)
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, n_classes)
    one_hot[np.arange(batch_size), labels] = 1
    return one_hot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--load_path', default=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--data_file', default='test_files')
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--epoch', default=None)
    parser.add_argument('--data_format', default='vctk')
    args = parser.parse_args()
    return args

def parse_fn(filename, dataset_format):
    if dataset_format == 'vctk':
        src_spk, phrase_id = re.match(r'(\S+)_(\d+).wav',os.path.basename(filename)).groups()
    elif dataset_format == 'alcaim':
        src_spk, phrase_id = re.match(r'(\S+)-(\d+).wav',os.path.basename(filename)).groups()
    elif dataset_format == 'smt':
        phrase_id = re.match(r'list(\S+).wav',os.path.basename(filename)).groups()
    else:
        phrase_id = os.path.splitext(file_name)[0]
        
    return phrase_id

def generate_signals(save_path, data_path, load_path, config_file = None, data_file = 'test_files', epoch = None, dataset_format = 'vctk'):
    
    save_path = Path(save_path)
    data_path = Path(data_path)
    load_path = Path(load_path)
    
    if config_file != None:
        hp = HParam(config_file)
    else:
        hp = HParam(load_path / 'config.yaml')
        
    os.makedirs(save_path, exist_ok=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    test_dataset = dataset.WaveDataset(data_path / data_file, data_path / 'speakers', sample_rate=hp.model.sample_rate)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                   batch_size=1,
                                   num_workers=1,
                                   collate_fn=dataset.collate_fn,
                                   shuffle=False,pin_memory=True,
                                   normalization_db=-25)
    
    nl = hp.model.generator.norm_layer
    wn = hp.model.generator.weight_norm
    cond = hp.model.generator.conditioning
    G = Generator(hp.model.generator.decoder_ratios,
                  hp.model.generator.decoder_channels,
                  hp.model.generator.num_bottleneck_layers,
                  test_dataset.num_spk, 
                  hp.model.generator.conditional_dim,
                  norm_layer = (nl.bottleneck, nl.encoder, nl.decoder),
                  weight_norm = (wn.bottleneck, wn.encoder, wn.decoder),
                  bot_cond = cond.bottleneck, enc_cond = cond.encoder, dec_cond = cond.decoder).to(device)
    
    g_file = 'step{}-G.pt'.format(epoch) if epoch != None else 'latest-G.pt'
    print('Loading from {}'.format(load_path / g_file))
    G.load_state_dict(torch.load(load_path / g_file, map_location=lambda storage, loc: storage))
    
    ds_spks = []
    for i, data in enumerate(test_data_loader):
        signal_real, label_src = data
        if label_src not in ds_spks:
            ds_spks.append(label_src.item())
    #print(ds_spks)
    #print(test_data_loader.dataset.dataset)
    
    for i, data in tqdm(enumerate(test_data_loader)):
        signal_real, label_src = data
        c_src = label2onehot(label_src,test_dataset.num_spk)
        signal_real = signal_real.to(device)
        c_src = c_src.to(device)
        label_src = label_src.item()
        file_name = test_data_loader.dataset.get_filename(i)
        #base_name = os.path.splitext(file_name)[0]
        phrase_id = parse_fn(file_name, dataset_format)
        
        #print(i, file_name, label_src)
        #print(type(signal_real))
        
        for tgt_spk in ds_spks:
            label_tgt = torch.tensor([tgt_spk])
            c_tgt = label2onehot(label_tgt,test_dataset.num_spk)
            #print(c_src, c_tgt)
            label_tgt = label_tgt.item()
            
            c_tgt = c_tgt.to(device)
            
            signal_fake = G(signal_real,c_tgt,c_src)
            
            signal_fake = signal_fake.squeeze().cpu().detach().numpy()
            
            #sf.write(save_path / 'sig{:02d}_{:1d}-{:1d}_conv.wav'.format(i,label_src,label_tgt),signal_fake,hp.model.sample_rate)
            #sf.write(save_path / '{}_{:1d}-{:1d}_conv.wav'.format(base_name,label_src,label_tgt),signal_fake,hp.model.sample_rate)
            sf.write(save_path / '{}-{}-{}-conv.wav'.format(phrase_id,test_dataset.spk_reverse_dict[label_src],test_dataset.spk_reverse_dict[label_tgt]),signal_fake,hp.model.sample_rate)
            
        signal_real = signal_real.squeeze().cpu().detach().numpy()
        #sf.write(save_path / 'sig{:02d}_{:1d}-X_orig.wav'.format(i,label_src),signal_real,hp.model.sample_rate)
        sf.write(save_path / '{}-{}-X-orig.wav'.format(phrase_id,test_dataset.spk_reverse_dict[label_src]),signal_real,hp.model.sample_rate)

if __name__ == '__main__':
    args = parse_args()
    generate_signals(args.save_path, args.data_path, args.load_path, args.config_file, args.data_file, args.epoch, args.data_format)
