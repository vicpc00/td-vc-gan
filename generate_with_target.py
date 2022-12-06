import os
import argparse
from pathlib import Path

import soundfile as sf

import torch
import numpy as np
from tqdm import tqdm

from model.generator import Generator
import data.dataset as dataset

import util
import util.yin as torchyin

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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_path = Path(args.save_path)
    data_path = Path(args.data_path)
    load_path = Path(args.load_path)
    
    if args.config_file != None:
        hp = HParam(args.config_file)
    else:
        hp = HParam(load_path / 'config.yaml')
        
    os.makedirs(save_path, exist_ok=True)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    test_dataset = dataset.WaveDataset(data_path / args.data_file, data_path / 'speakers', sample_rate=hp.model.sample_rate, add_new_spks=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                   batch_size=1,
                                   num_workers=1,
                                   collate_fn=dataset.collate_fn,
                                   shuffle=False,pin_memory=True)
    
    ds_spks = [] #Speakers that are actually in the dataset
    for i, data in enumerate(test_data_loader):
        signal_real, label_src = data
        if label_src not in ds_spks:
            ds_spks.append(label_src.item())
    #print(ds_spks)
    #print(test_data_loader.dataset.spk_dict)
    #print(test_dataset.spk_reverse_dict)
    #print(test_data_loader.dataset.dataset)
    
    speaker_datasets = {}
    speaker_dataloaders = {}
    for spk_id in ds_spks:
        spk_label = test_dataset.spk_reverse_dict[spk_id]
        speaker_dataset = dataset.SpeakerDataset(spk_label, data_path / args.data_file, data_path / 'speakers', sample_rate=hp.model.sample_rate, add_new_spks=True)
        speaker_dataloader = torch.utils.data.DataLoader(speaker_dataset,
                                                            batch_size=1,
                                                            num_workers=1,
                                                            collate_fn=dataset.collate_fn,
                                                            shuffle=True,pin_memory=True)
        speaker_dataloaders[spk_id] = speaker_dataloader
    
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
    
    g_file = 'step{}-G.pt'.format(args.epoch) if args.epoch != None else 'latest-G.pt'
    print('Loading from {}'.format(load_path / g_file))
    G.load_state_dict(torch.load(load_path / g_file, map_location=lambda storage, loc: storage))
    
    for i, data in tqdm(enumerate(test_data_loader)):
        
        speaker_dataloader_iters = {spk_id: iter(dl) for spk_id, dl in speaker_dataloaders.items()}
        
        signal_real, label_src = data
        c_src = label2onehot(label_src,test_dataset.num_spk)
        signal_real = signal_real.to(device)
        c_src = c_src.to(device)
        label_src = label_src.item()
        file_name = test_data_loader.dataset.get_filename(i)
        base_name = os.path.splitext(file_name)[0]
        #print(i, file_name, label_src)
        #print(type(signal_real))
        
        f0_src = torchyin.estimate(signal_real, sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)
        mu_src = torch.sum((f0_src>0)*torch.log(f0_src+1e-6), -1, keepdim=True)/(torch.sum(f0_src>0, -1, keepdim=True)+1e-6)
        
        
        for tgt_spk in ds_spks:
            signal_tgt, label_tgt = next(speaker_dataloader_iters[tgt_spk])
            
            #label_tgt = torch.tensor([tgt_spk])
            c_tgt = label2onehot(label_tgt,test_dataset.num_spk)
            #print(c_src, c_tgt)
            label_tgt = label_tgt.item()
            
            signal_tgt = signal_tgt.to(device)
            c_tgt = c_tgt.to(device)
            
            
            f0_tgt = torchyin.estimate(signal_tgt, sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)

            mu_tgt = torch.sum((f0_tgt>0)*torch.log(f0_tgt+1e-6), -1, keepdim=True)/(torch.sum(f0_tgt>0, -1, keepdim=True)+1e-6)
            f0_conv_tgt = torch.zeros(f0_src.shape).to(device)
            f0_conv_tgt[f0_src>0] = torch.exp(torch.log(f0_src+1e-6) + mu_tgt - mu_src)[f0_src>0]
            
            c_f0_conv = util.f0_to_excitation(f0_conv_tgt, 64, sampling_rate=hp.model.sample_rate)
            
            signal_fake = G(signal_real,c_tgt, c_var=c_f0_conv)
            
            signal_fake = signal_fake.squeeze().cpu().detach().numpy()
            
            #sf.write(save_path / 'sig{:02d}_{:1d}-{:1d}_conv.wav'.format(i,label_src,label_tgt),signal_fake,hp.model.sample_rate)
            #sf.write(save_path / '{}_{:1d}-{:1d}_conv.wav'.format(base_name,label_src,label_tgt),signal_fake,hp.model.sample_rate)
            sf.write(save_path / '{}_{}-{}_conv.wav'.format(base_name,test_dataset.spk_reverse_dict[label_src],test_dataset.spk_reverse_dict[label_tgt]),signal_fake,hp.model.sample_rate)
            
        signal_real = signal_real.squeeze().cpu().detach().numpy()
        #sf.write(save_path / 'sig{:02d}_{:1d}-X_orig.wav'.format(i,label_src),signal_real,hp.model.sample_rate)
        sf.write(save_path / '{}_{}-X_orig.wav'.format(base_name,test_dataset.spk_reverse_dict[label_src]),signal_real,hp.model.sample_rate)

if __name__ == '__main__':
    main()
