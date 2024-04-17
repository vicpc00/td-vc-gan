import os
import argparse
from pathlib import Path
import re

import soundfile as sf

import torch
import numpy as np
from tqdm import tqdm

from model.generator import Generator
from data.dataset import collate_fn
from data.pairs_dataset import PairsDataset

import util
import util.yin as torchyin
import util.crepe


from util.hparams import HParam


from generate_with_target import label2onehot, parse_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--load_path', default=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--data_file', default='test_files')
    parser.add_argument('--conv_file', required=True)
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--epoch', default=None)
    parser.add_argument('--data_format', default='vctk')
    args = parser.parse_args()
    return args

def generate_signals(save_path, data_path, load_path, conv_file,
                     config_file = None, data_file = 'test_files', 
                     epoch = None, dataset_format = 'vctk'):

    save_path = Path(save_path)
    data_path = Path(data_path)
    load_path = Path(load_path)
    
    if args.config_file != None:
        hp = HParam(args.config_file)
    else:
        hp = HParam(load_path / 'config.yaml')

    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = PairsDataset(conv_file, data_path / args.data_file, data_path / 'speakers', 
                                sample_rate=hp.model.sample_rate, return_index = False, 
                                normalization_db=hp.train.normalization_db)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                   batch_size=1,
                                   num_workers=8,
                                   collate_fn=collate_fn,
                                   shuffle=False,pin_memory=True)

    nl = hp.model.generator.norm_layer
    wn = hp.model.generator.weight_norm
    cond = hp.model.generator.conditioning
    G = Generator(hp.model.generator.decoder_ratios,
                  hp.model.generator.decoder_channels,
                  hp.model.generator.num_bottleneck_layers,
                  test_dataset.num_spk, 
                  hp.model.generator.conditional_dim,
                  hp.model.generator.content_dim,
                  hp.model.generator.num_res_blocks,
                  hp.model.generator.num_enc_layers,
                  hp.model.generator.encoder_model,
                  norm_layer = (nl.bottleneck, nl.encoder, nl.decoder),
                  weight_norm = (wn.bottleneck, wn.encoder, wn.decoder),
                  bot_cond = cond.bottleneck, enc_cond = cond.encoder, dec_cond = cond.decoder).to(device)
    
    g_file = 'step{}-G.pt'.format(args.epoch) if args.epoch != None else 'latest-G.pt'
    print('Loading from {}'.format(load_path / g_file))
    G.load_state_dict(torch.load(load_path / g_file, map_location=lambda storage, loc: storage))
    
    decoder = "argmax"

    for i, data in tqdm(enumerate(test_data_loader)):
        
        signal_real, label_src, signal_tgt, label_tgt = data

        signal_real = signal_real.to(device)
        c_src = label2onehot(label_src,test_dataset.num_spk)
        c_src = c_src.to(device)
        signal_tgt = signal_tgt.to(device)
        c_tgt = label2onehot(label_tgt,test_dataset.num_spk)
        c_tgt = c_tgt.to(device)

        conv_name = test_dataset.get_convname(i)
        #print(conv_name)

        f0_src, _ = util.crepe.filtered_pitch(signal_real, decoder)
        mu_src = torch.sum((f0_src>0)*torch.log(f0_src+1e-6), -1, keepdim=True)/(torch.sum(f0_src>0, -1, keepdim=True)+1e-6)
        
        f0_tgt, _ = util.crepe.filtered_pitch(signal_tgt, decoder)
        mu_tgt = torch.sum((f0_tgt>0)*torch.log(f0_tgt+1e-6), -1, keepdim=True)/(torch.sum(f0_tgt>0, -1, keepdim=True)+1e-6)
        
        f0_conv_tgt = torch.zeros(f0_src.shape).to(device)
        f0_conv_tgt[f0_src>0] = torch.exp(torch.log(f0_src+1e-6) + mu_tgt - mu_src)[f0_src>0]
        
        c_f0_conv = util.f0_to_excitation(f0_conv_tgt, 64, sampling_rate=hp.model.sample_rate)
        
        signal_fake = G(signal_real,c_tgt, c_var=c_f0_conv)
        signal_fake = signal_fake.squeeze().cpu().detach().numpy()

        sf.write(save_path / f'{conv_name}.wav',signal_fake,hp.model.sample_rate)

if __name__ == '__main__':
    args = parse_args()
    generate_signals(args.save_path, args.data_path, args.load_path, args.conv_file, args.config_file, args.data_file, args.epoch, args.data_format)
