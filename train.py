
import os
import sys
import shutil
import argparse
from pathlib import Path
import subprocess
import pickle

import soundfile as sf

import torch
import torch.nn.functional as F
import numpy as np

import util
import util.yin as torchyin
import util.crepe

import torch.utils.tensorboard as tensorboard

from model.generator import Generator
from model.discriminator import MultiscaleDiscriminator
from model.latent_classifier import LatentClassifier
from model.f0_estimator import F0Estimator
import data.dataset as dataset

import util
import util.audio
from util.hparams import HParam

import util.losses
#torch.autograd.set_detect_anomaly(False)

#import timeit


def label2onehot(labels, n_classes):
    #labels: (batch_size,)
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, n_classes)
    one_hot[np.arange(batch_size), labels] = 1
    return one_hot
def onehot2label(one_hot, dataset):
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--config_file', default='./config/default.yaml')
    parser.add_argument('--epoch', default=None)
    args = parser.parse_args()
    return args

def load_model(model, path):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print(f'Warning: default loading for {path} failed. Trying permisive load')
        messages = util.load_possible(model, state_dict)
        for msg_type in messages:
            if msg_type in ['matched']:
                continue
            for msg in messages[msg_type]:
                print(f'{msg_type}: {msg}')
        

#Random seed updater for workers. 
#See https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    initial_seed = 1234
    np.random.seed(initial_seed)
    
    args = parse_args()
    save_path = Path(args.save_path)
    data_path = Path(args.data_path)
    load_path = Path(args.load_path) if args.load_path else None
    hp = HParam(args.config_file)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path / 'generated', exist_ok=True)
    
    if args.epoch != None:
        shutil.copy2(args.config_file, save_path / 'config-epoch{}.yaml'.format(args.epoch))
    else:
        try:
            shutil.copy2(args.config_file, save_path / 'config.yaml')
        except shutil.SameFileError:
            pass
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    with open(save_path / 'githash','w') as f:
        f.write(message.strip().decode('utf-8'))
    with open(save_path / 'argv','w') as f:
        f.write(' '.join(sys.argv))
    logger = tensorboard.SummaryWriter(str(save_path / 'logs'))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset = dataset.WaveDataset(data_path / 'train_files', data_path / 'speakers', sample_rate=hp.model.sample_rate, 
                                        max_segment_size = hp.train.max_segment, augment_noise = 1e-9, 
                                        normalization_db = hp.train.normalization_db, data_augment = True)
    test_dataset = dataset.WaveDataset(data_path / 'test_files', data_path / 'speakers', sample_rate=hp.model.sample_rate,
                                        max_segment_size = hp.test.max_segment, normalization_db = hp.train.normalization_db)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                   batch_size=hp.train.batch_size,
                                   num_workers=int(hp.train.num_workers),
                                   collate_fn=dataset.collate_fn,
                                   shuffle=True,pin_memory=True,
                                   worker_init_fn=worker_init_fn)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                   batch_size=1,
                                   num_workers=1,
                                   collate_fn=dataset.collate_fn,
                                   shuffle=True,pin_memory=True,
                                   worker_init_fn=worker_init_fn)
    val_data_loader = test_data_loader
    nl = hp.model.generator.norm_layer
    wn = hp.model.generator.weight_norm
    cond = hp.model.generator.conditioning
    latent_classier = True
    G = Generator(hp.model.generator.decoder_ratios,
                  hp.model.generator.decoder_channels,
                  hp.model.generator.num_bottleneck_layers,
                  train_dataset.num_spk, 
                  hp.model.generator.conditional_dim,
                  hp.model.generator.content_dim,
                  hp.model.generator.num_res_blocks,
                  norm_layer = (nl.bottleneck, nl.encoder, nl.decoder),
                  weight_norm = (wn.bottleneck, wn.encoder, wn.decoder),
                  bot_cond = cond.bottleneck, enc_cond = cond.encoder, dec_cond = cond.decoder,
                  output_content_emb = latent_classier)
    D = MultiscaleDiscriminator(hp.model.discriminator.num_disc,
                                train_dataset.num_spk,
                                hp.model.discriminator.num_layers,
                                hp.model.discriminator.num_channels_base,
                                hp.model.discriminator.num_channel_mult,
                                hp.model.discriminator.downsampling_factor,
                                hp.model.discriminator.conditional_dim,
                                hp.model.discriminator.conditional_spks)

    if hp.train.lambda_latcls != 0 or hp.log.val_lat_cls:
        C = LatentClassifier(train_dataset.num_spk,hp.model.generator.decoder_channels[0])
 
    if load_path != None:
        if args.epoch != None:
            load_file_base = 'step{}'.format(args.epoch)
            start_epoch = int(args.epoch) +1
        else:
            load_file_base = 'latest'
            start_epoch = 0
            """
            with open(save_path / 'latest_epoch','r') as f:
                start_epoch = int(f.read())
            """
            
        print('Loading from {}'.format(load_path / '{}-G.pt'.format(load_file_base)))

        load_model(G, load_path / '{}-G.pt'.format(load_file_base))
        load_model(D, load_path / '{}-D.pt'.format(load_file_base))
        #G.load_state_dict(torch.load(load_path / '{}-G.pt'.format(load_file_base), map_location=lambda storage, loc: storage))
        #D.load_state_dict(torch.load(load_path / '{}-D.pt'.format(load_file_base), map_location=lambda storage, loc: storage))

        if 'C' in locals():
            if os.path.exists(load_path / '{}-C.pt'.format(load_file_base)):
                load_model(C, load_path / '{}-C.pt'.format(load_file_base))
            #C.load_state_dict(torch.load(load_path / '{}-C.pt'.format(load_file_base), map_location=lambda storage, loc: storage))
            
    else:
        start_epoch = 0
    #print(G)
        
    #Send models to device
    G.to(device)
    D.to(device)       

    optimizer_G = torch.optim.Adam(G.parameters(), hp.train.lr_g, hp.train.adam_beta)
    optimizer_D = torch.optim.Adam(D.parameters(), hp.train.lr_d, hp.train.adam_beta)

    if 'C' in locals():
        optimizer_C = torch.optim.Adam(C.parameters(), hp.train.lr_d, hp.train.adam_beta)
        C.to(device)

    if hp.train.freeze_subnets is not None and 'encoder' in hp.train.freeze_subnets:
        for param in G.encoder.parameters():
            param.requires_grad = False

    need_target_signal = hp.train.lambda_f0 != 0
    
    f0_means = torch.zeros(train_dataset.num_spk).to(device)
    f0_Ns = torch.zeros(train_dataset.num_spk).to(device)

    #require: model, data_loader, dataset, num_epoch, start_epoch=0
    #Train Loop
    iter_count = 0
    for epoch in range(start_epoch, hp.train.num_epoch+1):
        for i, data in enumerate(train_data_loader):

            loss = {}

            #Real data
            signal_real, label_src = data
            c_src = label2onehot(label_src,train_dataset.num_spk)
            if hp.train.no_conv:
                label_tgt = label_src
                signal_real_tgt = signal_real
            else:
                if need_target_signal:
                    perm = np.random.permutation(signal_real.shape[0])
                    signal_real_tgt, label_tgt = signal_real[perm], label_src[perm]
                else:
                    #Random target label
                    label_tgt = torch.randint(train_dataset.num_spk,label_src.shape)
            c_tgt = label2onehot(label_tgt,train_dataset.num_spk)
            
            #Send everything to device
            signal_real = signal_real.to(device)
            label_src = label_src.to(device)
            label_tgt = label_tgt.to(device)
            c_src = c_src.to(device)
            c_tgt = c_tgt.to(device)
            if need_target_signal:
                signal_real_tgt = signal_real_tgt.to(device)
            
            #f0_src = torchyin.estimate(signal_real, sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)
            f0_src, f0_src_activ = util.crepe.filtered_pitch(signal_real)

            if hp.train.no_conv:
                f0_conv_tgt = f0_src
                f0_conv_tgt_activ = f0_src_activ
            else:
                f0_tgt = f0_src[perm]
                
                mu_tgt = torch.sum((f0_tgt>0)*torch.log(f0_tgt+1e-6), -1, keepdim=True)/(torch.sum(f0_tgt>0, -1, keepdim=True)+1e-6)
                mu_src = torch.sum((f0_src>0)*torch.log(f0_src+1e-6), -1, keepdim=True)/(torch.sum(f0_src>0, -1, keepdim=True)+1e-6)
                
                f0_conv_tgt = torch.zeros(f0_src.shape).to(device)
                f0_conv_tgt[f0_src>0] = torch.exp(torch.log(f0_src+1e-6) + mu_tgt - mu_src)[f0_src>0]
                f0_conv_tgt_activ = util.roll_batches(f0_src_activ, util.crepe.get_shift(torch.exp(mu_src), torch.exp(mu_tgt)), 1)
                
            
            c_f0_conv = util.f0_to_excitation(f0_conv_tgt, 64, sampling_rate=hp.model.sample_rate)
            c_f0_src = util.f0_to_excitation(f0_src, 64, sampling_rate=hp.model.sample_rate)
            
            #Discriminator training
            if iter_count % hp.train.D_step_interval == 0:
                
                #Compute fake signal
                signal_fake = G(signal_real, c_tgt, c_var = c_f0_conv)
                sig_real_cont_emb = G.content_embedding.clone()
                
                #Real signal losses
                out_adv_real_list, features_real_list = D(signal_real, label_src)
                #Fake signal losses
                out_adv_fake_list, features_fake_list = D(signal_fake.detach(), label_tgt)
                
                d_loss_adv_real = 0
                d_loss_adv_fake = 0
                for i, (out_adv_fake, out_adv_real) in enumerate(zip(out_adv_fake_list,out_adv_real_list)):
                    d_loss_adv_real_ = F.mse_loss(out_adv_real,torch.ones(out_adv_real.size(), device=device))
                    d_loss_adv_fake_ = F.mse_loss(out_adv_fake,torch.zeros(out_adv_fake.size(), device=device))
                    #loss[f'D_loss_adv_real_{i}'] = d_loss_adv_real_.item()
                    #loss[f'D_loss_adv_fake_{i}'] = d_loss_adv_fake_.item()
                    d_loss_adv_real += d_loss_adv_real_
                    d_loss_adv_fake += d_loss_adv_fake_
                    
                d_gan_loss = d_loss_adv_real + d_loss_adv_fake
                
                
                #Full loss
                d_loss = d_gan_loss
                #Optimize
                optimizer_D.zero_grad()
                d_loss.backward()
                if hp.train.grad_max_norm_D is not None:
                    torch.nn.utils.clip_grad_norm_(D.parameters(),hp.train.grad_max_norm_D)
                optimizer_D.step()
    
                #Logging
                loss['D_loss_adv_real'] = d_loss_adv_real.item()
                loss['D_loss_adv_fake'] = d_loss_adv_fake.item()
                loss['D_loss'] = d_loss.item()
            

                #Latent classifier step
                if hp.train.lambda_latcls != 0 or hp.log.val_lat_cls:
                    out_lat_cls = C(sig_real_cont_emb)
                    c_loss = F.cross_entropy(out_lat_cls,label_src)
                    
                    optimizer_C.zero_grad()
                    c_loss.backward()
                    optimizer_C.step()
                
                    loss['C_loss'] = c_loss.item()
                    loss['C_acc'] = torch.sum(torch.argmax(out_lat_cls,dim=1) == label_src)/hp.train.batch_size
                
                del out_adv_real_list
                del out_adv_fake_list, features_fake_list
                del out_adv_fake, out_adv_real, d_gan_loss
                del d_loss
                if hp.train.lambda_latcls != 0:
                    del out_lat_cls, c_loss      


            #Generator training
            if iter_count % hp.train.G_step_interval == 0: #N steps of D for each steap of G
                #Fake signal losses
                signal_fake = G(signal_real, c_tgt, c_var = c_f0_conv)
                sig_real_cont_emb = G.content_embedding.clone()
                out_adv_fake_list, features_fake_list = D(signal_fake, label_tgt)

                g_loss_adv_fake = torch.zeros(1, device=device)
                for i, out_adv_fake in enumerate(out_adv_fake_list):
                    g_loss_adv_fake_ = F.mse_loss(out_adv_fake,torch.ones(out_adv_fake.size(), device=device))
                    #loss[f'G_loss_adv_fake_{i}'] = g_loss_adv_fake_
                    g_loss_adv_fake += g_loss_adv_fake_
                    
                if hp.train.lambda_rec > 0 or hp.train.lambda_idt > 0:
                    #Real signal losses
                    if hp.train.jitter_amp > 0:
                        signal_real_jitter = util.audio.add_jitter(signal_real, hp.train.jitter_amp)
                    else:
                        signal_real_jitter = signal_real
                    if hp.train.lambda_feat > 0:
                        _, features_real_list = D(signal_real_jitter, label_src)
                    
                g_loss_rec = torch.zeros(1, device=device)
                if not hp.train.no_conv and hp.train.lambda_rec > 0:
                    #Reconstructed signal losses
                    signal_rec = G(signal_fake, c_src, c_var = c_f0_src)
                    
                    sig_fake_cont_emb = G.content_embedding.clone()
                    if hp.train.lambda_feat > 0:
                        _, features_rec_list = D(signal_rec, label_src)
                        g_loss_rec_feat = util.losses.multiscale_feat_loss(features_rec_list, features_real_list ,norm_p = 1)
                        g_loss_rec += g_loss_rec_feat
                        loss['G_loss_rec_feat'] = g_loss_rec_feat
                    if hp.train.lambda_spec > 0:
                        g_loss_rec_spec = util.losses.multiscale_spec_loss(signal_rec, signal_real_jitter, [2048, 1024, 512])
                        g_loss_rec += g_loss_rec_spec
                        loss['G_loss_rec_spec'] = g_loss_rec_spec
                    if hp.train.lambda_wave > 0:
                        g_loss_rec_wave = torch.mean(torch.abs(signal_real - signal_rec))#L1 Loss
                        g_loss_rec += g_loss_rec_wave
                        loss['G_loss_rec_wave'] = g_loss_rec_wave

                #Identity loss
                g_loss_idt = torch.zeros(1, device=device)
                if hp.train.lambda_idt > 0:
                    if not hp.train.no_conv:
                        signal_idt = G(signal_real, c_src, c_var = c_f0_src)

                    else:
                        signal_idt = signal_fake
                        
                    if hp.train.lambda_feat > 0:
                        _, features_idt_list = D(signal_idt, label_src)
                        g_loss_idt_feat = util.losses.multiscale_feat_loss(features_idt_list, features_real_list ,norm_p = 1)
                        g_loss_idt += g_loss_idt_feat
                        loss['G_loss_idt_feat'] = g_loss_idt_feat
                    if hp.train.lambda_spec > 0:
                        g_loss_idt_spec = util.losses.multiscale_spec_loss(signal_idt, signal_real_jitter, [2048, 1024, 512])
                        g_loss_idt += g_loss_idt_spec
                        loss['G_loss_idt_spec'] = g_loss_idt_spec
                    if hp.train.lambda_wave > 0:
                        g_loss_idt_wave = torch.mean(torch.abs(signal_real - signal_idt))#L1 Loss
                        g_loss_rec += g_loss_idt_wave
                        loss['G_loss_idt_wave'] = g_loss_idt_wave
                

                #Content embedding loss
                if hp.train.lambda_cont_emb > 0:
                    if hp.train.lambda_rec == 0:    
                        sig_fake_cont_emb = G.encoder(signal_fake)
                    g_loss_cont_emb = torch.mean(torch.abs(sig_fake_cont_emb - sig_real_cont_emb))
                else:
                    g_loss_cont_emb = torch.zeros(1, device=device)

                    
                #Latent classification loss
                if hp.train.lambda_latcls != 0:
                    out_lat_cls = C(sig_real_cont_emb)
                    g_loss_lat_cls = F.cross_entropy(out_lat_cls,label_src)
                else:

                    g_loss_lat_cls = torch.zeros(1, device=device)

                    
                #F0 loss
                if hp.train.lambda_f0 != 0:
                    #f0_tgt = torchyin.estimate(signal_real_tgt.cpu(), sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)
                    #f0_conv, voiced_conv = f0_est(signal_fake)
                    # if epoch == start_epoch:
                    #     f0_tgt_mean = torch.sum((f0_tgt>0)*(f0_tgt), (-2,-1))/(torch.sum(f0_tgt>0, (-2,-1))+1e-6)
                    #     alpha = f0_Ns[label_tgt]/(f0_Ns[label_tgt]+1)
                    #     f0_means[label_tgt] = alpha*f0_Ns[label_tgt] + (1-alpha)*f0_tgt_mean.detach()
                    #     f0_Ns[label_tgt] = f0_Ns[label_tgt]+1
                    
                    #f0_conv = torchyin.estimate(signal_fake.cpu(), sample_rate=hp.model.sample_rate, frame_stride=64/16000, soft = True).to(device)
                    f0_conv, f0_conv_activ = util.crepe.filtered_pitch(signal_fake)

                    #f0_src, f0_tgt, f0_conv, f0_conv_tgt = f0_src/400, f0_tgt/400, f0_conv/400, f0_conv_tgt/400

                    if False:
                        #g_loss_f0 = torch.abs(torch.mean(f0_tgt[f0_tgt>0],-1) - torch.mean(f0_conv[voiced_conv>.5],-1))
                        #g_loss_f0 = torch.pow(torch.mean(torch.log(f0_tgt[f0_tgt>0]),-1) - torch.mean(torch.log(f0_conv[f0_conv>0]),-1), 2)
                        #g_loss_f0 = torch.pow(torch.mean(f0_tgt[f0_tgt>0],-1) - torch.mean(f0_conv[f0_conv>0],-1), 2)
                        g_loss_f0 = torch.pow(torch.sum((f0_tgt>0)*(f0_tgt), -1, keepdim=True)/(torch.sum(f0_tgt>0, -1, keepdim=True)+1e-6) -
                                              torch.sum((f0_conv>0)*(f0_conv), -1, keepdim=True)/(torch.sum(f0_conv>0, -1, keepdim=True)+1e-6),2)
                        g_loss_f0[g_loss_f0.isnan().detach()] = 0
                        g_loss_f0 = torch.mean(g_loss_f0)

                    else:
                        #f0_src = torchyin.estimate(signal_real.cpu(), sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)
                        #mu_tgt = torch.mean(torch.log(f0_tgt[f0_tgt>0]),-1)
                        #mu_src = torch.mean(torch.log(f0_src[f0_src>0]),-1)
                        #mu_tgt = torch.nanmean(torch.log(torch.where(f0_tgt>0, f0_tgt, torch.nan)),-1, keepdim=True)
                        #mu_src = torch.nanmean(torch.log(torch.where(f0_src>0, f0_src, torch.nan)),-1, keepdim=True)
                        #mu_conv = torch.sum((f0_conv>0)*(f0_conv), -1, keepdim=True)/torch.sum(f0_conv>0, -1, keepdim=True)
                        #mu_tgt = f0_means[label_tgt].view(mu_conv.shape)
                        
                        #f0_conv_alt = torch.zeros(f0_src.shape).to(device)
                        #f0_conv_alt[f0_conv>0] = (f0_conv - mu_conv + mu_tgt)[f0_conv>0]
                        
                        #f0_conv_tgt = f0_src
                        
                        #g_loss_f0 = F.mse_loss(f0_conv[f0_conv>0],f0_conv_alt[f0_conv>0])
                        #g_loss_f0 = F.mse_loss(f0_conv,f0_conv_tgt)
                        #mask = (f0_src>0).expand(f0_conv_tgt_activ.shape)
                        #g_loss_f0 = F.mse_loss(f0_conv_activ[mask],f0_conv_tgt_activ[mask].detach())
                        g_loss_f0 = F.mse_loss(f0_conv_activ,f0_conv_tgt_activ.detach())
                        
                else:
                    g_loss_f0 = torch.zeros(1, device=device)
                    
                
                #Full loss
                g_loss = g_loss_adv_fake + \
                         hp.train.lambda_rec*g_loss_rec + \
                         hp.train.lambda_idt*g_loss_idt + \
                         hp.train.lambda_latcls*g_loss_lat_cls + \
                         hp.train.lambda_cont_emb*g_loss_cont_emb + \
                         hp.train.lambda_f0*g_loss_f0

                #Optimize
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                g_loss.backward()
                
                if hp.train.grad_max_norm_G is not None:
                    torch.nn.utils.clip_grad_norm_(G.parameters(),hp.train.grad_max_norm_G)
                optimizer_G.step()
                
                #Logging

                #loss['G_loss_adv_fake'] = g_loss_adv_fake if type(g_loss_adv_fake) == int else g_loss_adv_fake.item() #Check if int because it can be 0
                loss['G_loss_adv_fake'] = g_loss_adv_fake.item()
                loss['G_loss_rec'] = g_loss_rec.item()
                loss['G_loss_idt'] = g_loss_idt.item()
                loss['G_loss_lat_cls'] = g_loss_lat_cls.item()
                loss['G_loss_cont_emb'] = g_loss_cont_emb.item()
                loss['g_loss_f0'] = g_loss_f0.item()
         
                del out_adv_fake_list
                del out_adv_fake
                del g_loss_adv_fake
                if not hp.train.no_conv and hp.train.lambda_rec > 0:
                    del features_rec_list, g_loss_rec
                if hp.train.lambda_idt > 0:
                    del g_loss_idt, features_idt_list
                del g_loss
            
            #Print Losses
            if iter_count % hp.log.log_interval == 0:
                #print(torch.cuda.memory_summary(device=None, abbreviated=False))
                print('Epoch {}/{}, Itt {}'.format(epoch, hp.train.num_epoch, iter_count), end='')
                for label, value in loss.items():
                    #print(', {}: {}'.format(label, value))
                    logger.add_scalar(label,value,iter_count)
                    print(', {}: {:.4f}'.format(label, value),end='')
                print()
            iter_count += 1
            #break
            
        if epoch % hp.log.val_interval == 0:
            print('Validation loop')
            G.eval()
            D.eval()
            loss = {}
            with torch.no_grad():
                for i, data in enumerate(val_data_loader):
                    
                    signal_real, label_src = data
                    c_src = label2onehot(label_src,train_dataset.num_spk)
                    if hp.train.no_conv:
                        label_tgt = label_src
                    else:
                        #Random target label
                        label_tgt = torch.randint(train_dataset.num_spk,label_src.shape)
                    c_tgt = label2onehot(label_tgt,train_dataset.num_spk)

                    #Send everything to device
                    signal_real = signal_real.to(device)
                    label_src = label_src.to(device)
                    label_tgt = label_tgt.to(device)
                    c_src = c_src.to(device)
                    c_tgt = c_tgt.to(device)
                    
                    #f0_src = torchyin.estimate(signal_real, sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)
                    f0_src, f0_src_activ = util.crepe.filtered_pitch(signal_real)
                    c_f0 = util.f0_to_excitation(f0_src, 64, sampling_rate=hp.model.sample_rate)
                    
                    #Compute fake signal
                    signal_fake = G(signal_real, c_tgt, c_var = c_f0)
                    #Real signal losses
                    out_adv_real_list, features_real_list = D(signal_real, label_src)
                    #Fake signal losses
                    out_adv_fake_list, features_fake_list = D(signal_fake.detach(), label_tgt)
                    
                    d_loss_adv_real = 0
                    d_loss_adv_fake = 0
                    g_loss_adv_fake = 0
                    for out_adv_fake, out_adv_real in zip(out_adv_fake_list,out_adv_real_list):
                        d_loss_adv_real += F.mse_loss(out_adv_real,torch.ones(out_adv_real.size(), device=device))
                        d_loss_adv_fake += F.mse_loss(out_adv_fake,torch.zeros(out_adv_fake.size(), device=device))
                        g_loss_adv_fake += F.mse_loss(out_adv_fake,torch.ones(out_adv_fake.size(), device=device))
                    d_gan_loss = d_loss_adv_real + d_loss_adv_fake
                    
                    if 'C' in locals():
                        sig_cont_emb = G.content_embedding
                        out_lat_cls = C(sig_cont_emb)
                        g_loss_lat_cls = F.cross_entropy(out_lat_cls,label_src)
                        c_acc = torch.sum(torch.argmax(out_lat_cls,dim=1) == label_src)
                    else:
                        g_loss_lat_cls = torch.tensor([0])
                        c_acc = torch.tensor([0])
                    
                    d_loss = d_gan_loss
                    g_loss = g_loss_adv_fake
                    
                    loss['val_loss_adv_real'] = loss.setdefault('val_loss_adv_real',0) + d_loss_adv_real.item()
                    loss['val_loss_adv_fake'] = loss.setdefault('val_loss_adv_fake',0) + d_loss_adv_fake.item()
                    loss['val_loss_lat_cls'] = loss.setdefault('val_loss_lat_cls',0) + g_loss_lat_cls.item()
                    loss['val_D_loss'] = loss.setdefault('val_D_loss',0) + d_loss.item()
                    loss['val_G_loss'] = loss.setdefault('val_G_loss',0) + g_loss.item()
                    loss['val_C_acc'] = loss.setdefault('val_C_acc',0) + c_acc.item()
                    #break
                    
            print('Val Epoch {}/{}, Itt {}'.format(epoch, hp.train.num_epoch, iter_count), end='')
            for label, value in loss.items():
                logger.add_scalar(label,value/len(val_data_loader),iter_count)
                print(', {}: {:.4f}'.format(label, value/len(val_data_loader)),end='')
            print()
            G.train()
            D.train()
            
        #Save Model
        if epoch % hp.log.save_interval == 0:
            print('Saving checkpoint')
            torch.save(G.state_dict(), save_path / 'step{}-G.pt'.format(epoch))
            torch.save(D.state_dict(), save_path / 'step{}-D.pt'.format(epoch))
            torch.save(G.state_dict(), save_path / 'latest-G.pt')
            torch.save(D.state_dict(), save_path / 'latest-D.pt')
            if 'C' in locals():
                torch.save(C.state_dict(), save_path / 'step{}-C.pt'.format(epoch))
                torch.save(C.state_dict(), save_path / 'latest-C.pt')
            with open(save_path / 'latest_epoch','w') as f:
                f.write(str(epoch))
            print('Saved')
            
        #Gen exemples
        if epoch % hp.log.gen_interval == 0:
            print('Saving signals')
            f0_ratios = torch.rand(hp.log.gen_num)*1.5 + 0.5
            f0_ratios[0] = 1
            for i, data in enumerate(test_data_loader):
                if i >= hp.log.gen_num:
                    break
                signal_real, label_src = data
                c_src = label2onehot(label_src,train_dataset.num_spk)
                
                label_tgt = label_src if hp.train.no_conv else torch.randint(train_dataset.num_spk,label_src.shape)
                c_tgt = label2onehot(label_tgt,train_dataset.num_spk)
                
                signal_real = signal_real.to(device)
                label_src = label_src.item()
                label_tgt = label_tgt.item()
                c_src = c_src.to(device)
                c_tgt = c_tgt.to(device)
                
                #f0_src = torchyin.estimate(signal_real, sample_rate=hp.model.sample_rate, frame_stride=64/16000).to(device)
                f0_src, f0_src_activ = util.crepe.filtered_pitch(signal_real)
                c_f0 = util.f0_to_excitation(f0_src*f0_ratios[i], 64, sampling_rate=hp.model.sample_rate)
                
                signal_fake = G(signal_real,c_tgt,c_var = c_f0)
                signal_rec = G(signal_fake,c_src,c_var = c_f0)
                
                signal_real = signal_real.squeeze().cpu().detach().numpy()
                signal_fake = signal_fake.squeeze().cpu().detach().numpy()
                signal_rec  = signal_rec.squeeze().cpu().detach().numpy()
                
                sf.write(save_path / 'generated' / 'epoch{:03d}_sig{:02d}_{:1d}-{:1d}_conv_r={:.2f}.wav'.format(epoch,i,label_src,label_tgt, f0_ratios[i]),signal_fake,hp.model.sample_rate)
                sf.write(save_path / 'generated' / 'epoch{:03d}_sig{:02d}_{:1d}-{:1d}_orig.wav'.format(epoch,i,label_src,label_tgt),signal_real,hp.model.sample_rate)
                sf.write(save_path / 'generated' / 'epoch{:03d}_sig{:02d}_{:1d}-{:1d}_rec.wav'.format(epoch,i,label_src,label_tgt),signal_rec,hp.model.sample_rate)

        #Update random seed
        np.random.seed(initial_seed+epoch)


if __name__ == '__main__':
    main()
