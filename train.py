import os
import shutil
import argparse
from pathlib import Path

import soundfile as sf

import torch
import torch.nn.functional as F
import numpy as np

from model.generator import Generator
from model.discriminator import MultiscaleDiscriminator
import data.dataset as dataset

from util.hparams import HParam

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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_path = Path(args.save_path)
    data_path = Path(args.data_path)
    load_path = Path(args.load_path) if args.load_path else None
    hp = HParam(args.config_file)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path / 'generated', exist_ok=True)
    
    shutil.copy2(args.config_file, save_path / 'config.yaml')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = dataset.WaveDataset(data_path / 'train_files', data_path / 'speakers', sample_rate=hp.model.sample_rate, max_segment_size = hp.train.max_segment)
    test_dataset = dataset.WaveDataset(data_path / 'test_files', data_path / 'speakers', sample_rate=hp.model.sample_rate)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                   batch_size=hp.train.batch_size,
                                   num_workers=int(hp.train.num_workers),
                                   collate_fn=dataset.collate_fn,
                                   shuffle=True,pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                   batch_size=1,
                                   num_workers=1,
                                   collate_fn=dataset.collate_fn,
                                   shuffle=True,pin_memory=True)

    G = Generator(hp.model.generator.decoder_ratios,
                  hp.model.generator.decoder_channels,
                  hp.model.generator.num_bottleneck_layers,
                  train_dataset.num_spk, 
                  hp.model.generator.conditional_dim).to(device)
    D = MultiscaleDiscriminator(hp.model.discriminator.num_disc,
                                train_dataset.num_spk,
                                hp.model.discriminator.num_layers,
                                hp.model.discriminator.num_channels_base,
                                hp.model.discriminator.num_channel_mult,
                                hp.model.discriminator.downsampling_factor).to(device)

    if load_path != None:
        print('Loading from {}'.format(load_path / 'latest-G.pt'))
        G.load_state_dict(torch.load(load_path / 'latest-G.pt', map_location=lambda storage, loc: storage))
        D.load_state_dict(torch.load(load_path / 'latest-D.pt', map_location=lambda storage, loc: storage))

    optimizer_G = torch.optim.Adam(G.parameters(), hp.train.lr_g, hp.train.adam_beta)
    optimizer_D = torch.optim.Adam(D.parameters(), hp.train.lr_d, hp.train.adam_beta) 

    #require: model, data_loader, dataset, num_epoch, start_epoch=0
    #Train Loop
    iter_count = 0
    for epoch in range(hp.train.start_epoch, hp.train.num_epoch):
        for i, data in enumerate(train_data_loader):

            loss = {}

            #Real data
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

            #Compute fake signal
            signal_fake = G(signal_real,c_tgt)

            #Discriminator training
            #Real signal losses
            out_adv_real_list, out_cls_real_list, features_real_list = D(signal_real)
            #print(out_adv)
            #print(label_src.shape, out_cls_real_list[0].shape)
            d_loss_cls_real = 0
            for out_cls_real in out_cls_real_list:
                d_loss_cls_real += F.cross_entropy(out_cls_real, label_src)

            #Fake signal losses
            out_adv_fake_list, out_cls_fake_list, features_fake_list = D(signal_fake.detach())
            """
            d_loss_cls_fake = 0
            for out_cls_fake in out_cls_fake_list:
                d_loss_cls_fake += F.cross_entropy(out_cls_real, label_src)
            """

            #if hp.train.gan_loss == 'lsgan':
            d_loss_adv_real = 0
            d_loss_adv_fake = 0
            for out_adv_fake, out_adv_real in zip(out_adv_fake_list,out_adv_real_list):
                d_loss_adv_real += F.mse_loss(out_adv_real,torch.ones(out_adv_real.size()).to(device))
                d_loss_adv_fake += F.mse_loss(out_adv_fake,torch.zeros(out_adv_fake.size()).to(device))
            d_gan_loss = d_loss_adv_real + d_loss_adv_fake
            
            #Full loss
            d_loss = d_gan_loss + hp.train.lambda_cls*d_loss_cls_real
            #Optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            #Logging
            loss['D_loss_adv_real'] = d_loss_adv_real.item()
            loss['D_loss_adv_fake'] = d_loss_adv_fake.item()
            loss['D_loss_cls_real'] = d_loss_cls_real.item()

            #Generator training
            if iter_count % hp.train.D_to_G_train_ratio == 0: #N steps of D for each steap of G

                #Fake signal losses
                signal_fake = G(signal_real,c_tgt)
                out_adv_fake_list, out_cls_fake_list, _ = D(signal_fake)
                #if hp.train.gan_loss == 'lsgan':
                g_loss_adv_fake = 0
                g_loss_cls_fake = 0
                for out_adv_fake, out_cls_fake in zip(out_adv_fake_list, out_cls_fake_list):
                    g_loss_adv_fake += F.mse_loss(out_adv_fake,torch.ones(out_adv_fake.size()).to(device))
                    g_loss_cls_fake += F.cross_entropy(out_cls_fake, label_tgt)
                    
                if not hp.train.no_conv:
                    #Reconstructed signal losses
                    signal_rec = G(signal_fake, c_src)
                    if hp.train.rec_loss == 'feat':
                        _, _, features_rec_list = D(signal_rec)
                        g_loss_rec = 0
                        for features_rec, features_real in zip(features_rec_list, features_real_list):
                            for feat_rec, feat_real in zip(features_rec, features_real):
                                g_loss_rec += torch.mean(torch.abs(feat_rec - feat_real.detach()))#L1 Loss
                    else:
                        g_loss_rec = torch.mean(torch.abs(signal_real - signal_rec))#L1 Loss

                else:
                    g_loss_rec = 0

                    #Identity loss
                if hp.train.lambda_idt > 0:
                    if not hp.train.no_conv:
                        signal_idt = G(signal_real, c_src)
                    else:
                        signal_idt = signal_fake
                    if hp.train.rec_loss == 'feat':
                        _, _, features_idt_list = D(signal_idt)
                        g_loss_idt = 0
                        for features_idt, features_real in zip(features_idt_list, features_real_list):
                            for feat_idt, feat_real in zip(features_idt, features_real):
                                g_loss_idt += torch.mean(torch.abs(feat_idt - feat_real.detach()))#L1 Loss
                    else:
                        g_loss_idt = torch.mean(torch.abs(signal_real - signal_idt))#L1 Loss
                else:
                    g_loss_idt = 0
                    
                
                #Full loss
                g_loss = g_loss_adv_fake + hp.train.lambda_cls*g_loss_cls_fake + hp.train.lambda_rec*g_loss_rec + hp.train.lambda_rec*hp.train.lambda_idt*g_loss_idt
                #g_loss = g_loss_adv_fake + hp.train.lambda_cls*g_loss_cls_fake + hp.train.lambda_rec*g_loss_rec + hp.train.lambda_feat*g_loss_feat
                #Optimize
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
                
                #Logging
                loss['G_loss_adv_fake'] = g_loss_adv_fake if type(g_loss_adv_fake) == int else g_loss_adv_fake.item()
                loss['G_loss_cls_fake'] = g_loss_cls_fake if type(g_loss_cls_fake) == int else g_loss_cls_fake.item()
                loss['G_loss_rec'] = g_loss_rec if type(g_loss_rec) == int else g_loss_rec.item()
                loss['G_loss_idt'] = g_loss_idt if type(g_loss_idt) == int else g_loss_idt.item()

            #Print Losses
            if iter_count % hp.log.log_interval == 0:
                print('Epoch {}/{}, Iteration {}'.format(epoch, hp.train.num_epoch, iter_count), end='')
                for label, value in loss.items():
                    print(', {}: {:.4f}'.format(label, value),end='')
                print()
            iter_count += 1
        #Gen exemples
        if epoch % hp.log.gen_interval == 0:
            print('Saving signals')
            for i, data in enumerate(test_data_loader):
                if i >= hp.log.gen_num:
                    break
                signal_real, label_src = data
                c_src = label2onehot(label_src,train_dataset.num_spk)
                label_tgt = torch.randint(train_dataset.num_spk,label_src.shape)
                c_tgt = label2onehot(label_tgt,train_dataset.num_spk)
                
                signal_real = signal_real.to(device)
                label_src = label_src.item()
                label_tgt = label_tgt.item()
                c_src = c_src.to(device)
                c_tgt = c_tgt.to(device)
                
                signal_fake = G(signal_real,c_tgt)
                signal_rec = G(signal_fake,c_src)
                
                signal_real = signal_real.squeeze().cpu().detach().numpy()
                signal_fake = signal_fake.squeeze().cpu().detach().numpy()
                signal_rec  = signal_rec.squeeze().cpu().detach().numpy()
                
                sf.write(save_path / 'generated' / 'epoch{:03d}_sig{:02d}_{:1d}-{:1d}_conv.wav'.format(epoch,i,label_src,label_tgt),signal_fake,hp.model.sample_rate)
                sf.write(save_path / 'generated' / 'epoch{:03d}_sig{:02d}_{:1d}-{:1d}_orig.wav'.format(epoch,i,label_src,label_tgt),signal_real,hp.model.sample_rate)
                sf.write(save_path / 'generated' / 'epoch{:03d}_sig{:02d}_{:1d}-{:1d}_rec.wav'.format(epoch,i,label_src,label_tgt),signal_rec,hp.model.sample_rate)

                
        #Save Model
        if epoch % hp.log.save_interval == 0:
            print('Saving checkpoint')
            torch.save(G.state_dict(), save_path / 'step{}-G.pt'.format(epoch))
            torch.save(D.state_dict(), save_path / 'step{}-D.pt'.format(epoch))
            torch.save(G.state_dict(), save_path / 'latest-G.pt')
            torch.save(D.state_dict(), save_path / 'latest-D.pt')


if __name__ == '__main__':
    main()
