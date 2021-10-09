import os
import shutil
import argparse
from pathlib import Path
import subprocess
import pickle

import soundfile as sf

import torch
import torch.nn.functional as F
import numpy as np

import torch.utils.tensorboard as tensorboard

from model.generator import Generator
from model.discriminator import MultiscaleDiscriminator
from model.latent_classifier import LatentClassifier
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
    parser.add_argument('--epoch', default=None)
    args = parser.parse_args()
    return args

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
    logger = tensorboard.SummaryWriter(str(save_path / 'logs'))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset = dataset.WaveDataset(data_path / 'train_files', data_path / 'speakers', sample_rate=hp.model.sample_rate, 
                                        max_segment_size = hp.train.max_segment, augment_noise = 1e-9)
    test_dataset = dataset.WaveDataset(data_path / 'test_files', data_path / 'speakers', sample_rate=hp.model.sample_rate)

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
                  norm_layer = (nl.bottleneck, nl.encoder, nl.decoder),
                  weight_norm = (wn.bottleneck, wn.encoder, wn.decoder),
                  bot_cond = cond.bottleneck, enc_cond = cond.encoder, dec_cond = cond.decoder,
                  output_content_emb = latent_classier).to(device)
    D = MultiscaleDiscriminator(hp.model.discriminator.num_disc,
                                train_dataset.num_spk,
                                hp.model.discriminator.num_layers,
                                hp.model.discriminator.num_channels_base,
                                hp.model.discriminator.num_channel_mult,
                                hp.model.discriminator.downsampling_factor,
                                hp.model.discriminator.conditional_dim).to(device)
    C = LatentClassifier(train_dataset.num_spk,hp.model.generator.decoder_channels[0]).to(device)

    
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
        G.load_state_dict(torch.load(load_path / '{}-G.pt'.format(load_file_base), map_location=lambda storage, loc: storage))
        D.load_state_dict(torch.load(load_path / '{}-D.pt'.format(load_file_base), map_location=lambda storage, loc: storage))
    else:
        start_epoch = 0

    optimizer_G = torch.optim.Adam(G.parameters(), hp.train.lr_g, hp.train.adam_beta)
    optimizer_D = torch.optim.Adam(D.parameters(), hp.train.lr_d, hp.train.adam_beta) 
    optimizer_C = torch.optim.Adam(C.parameters(), hp.train.lr_d, hp.train.adam_beta) 

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
            signal_fake = G(signal_real,c_tgt,c_src)
            
            #Discriminator training
            #Real signal losses
            out_adv_real_list, out_cls_real_list, features_real_list = D(signal_real,c_src,c_tgt)
            #print(out_adv)
            #print(label_src.shape, out_cls_real_list[0].shape)
            """
            d_loss_cls_real = 0
            for out_cls_real in out_cls_real_list:
                #d_loss_cls_real += F.cross_entropy(out_cls_real, label_src)
                d_loss_cls_real += F.mse_loss(out_cls_real,torch.ones(out_cls_real.size()).to(device))
            """
            #Fake signal losses
            out_adv_fake_list, out_cls_fake_list, features_fake_list = D(signal_fake.detach(),c_tgt,c_src)
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
            
            d_loss_cls_real = 0
            d_loss_cls_fake = 0
            for out_cls_real,out_cls_fake in zip(out_cls_real_list,out_cls_fake_list):
                d_loss_cls_real += F.mse_loss(out_cls_real,torch.ones(out_cls_real.size()).to(device))
                d_loss_cls_fake += F.mse_loss(out_cls_fake,torch.zeros(out_cls_fake.size()).to(device))
            d_loss_cls = d_loss_cls_real+d_loss_cls_fake
            
            
            #Full loss
            d_loss = d_gan_loss + hp.train.lambda_cls*d_loss_cls
            #Optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            if hp.train.grad_max_norm_D is not None:
                torch.nn.utils.clip_grad_norm_(D.parameters(),hp.train.grad_max_norm_D)
            optimizer_D.step()

            #Logging
            loss['D_loss_adv_real'] = d_loss_adv_real.item()
            loss['D_loss_adv_fake'] = d_loss_adv_fake.item()
            loss['D_loss_cls_real'] = d_loss_cls_real.item()
            loss['D_loss_cls_fake'] = d_loss_cls_fake.item()
            
            #D_grad_norm = sum([param.grad.norm().item() for param in D.parameters()])
            D_grad_norm = torch.norm(torch.stack([param.grad.norm() for param in D.parameters()])).item()
            loss['D_loss_grad_norm'] = D_grad_norm
            
            #Latent classifier step
            sig_cont_emb = G.content_embedding
            out_lat_cls = C(sig_cont_emb)
            c_loss = F.cross_entropy(out_lat_cls,label_src)
            
            optimizer_C.zero_grad()
            c_loss.backward()
            optimizer_C.step()
            
            loss['C_loss'] = c_loss.item()
            loss['C_acc'] = torch.sum(torch.argmax(out_lat_cls,dim=1) == label_src)/hp.train.batch_size
            
            #del out_adv_real_list, out_cls_real_list
            #del out_adv_fake_list, out_cls_fake_list, features_fake_list
            #del out_adv_fake, out_adv_real, d_gan_loss
            #del out_cls_real, out_cls_fake, d_loss_cls
            #del d_loss
            

            #Generator training
            if iter_count % hp.train.D_to_G_train_ratio == 0: #N steps of D for each steap of G

                #Fake signal losses
                signal_fake = G(signal_real,c_tgt,c_src)
                out_adv_fake_list, out_cls_fake_list, _ = D(signal_fake,c_tgt,c_src)
                #if hp.train.gan_loss == 'lsgan':
                g_loss_adv_fake = 0
                g_loss_cls_fake = 0
                for out_adv_fake, out_cls_fake in zip(out_adv_fake_list, out_cls_fake_list):
                    g_loss_adv_fake += F.mse_loss(out_adv_fake,torch.ones(out_adv_fake.size()).to(device))
                    #g_loss_cls_fake += F.cross_entropy(out_cls_fake, label_tgt)
                    g_loss_cls_fake += F.mse_loss(out_cls_fake,torch.ones(out_cls_fake.size()).to(device))
                    
                if not hp.train.no_conv:
                    #Reconstructed signal losses
                    signal_rec = G(signal_fake, c_src, c_tgt)
                    if hp.train.rec_loss == 'feat':
                        _, _, features_rec_list = D(signal_rec,c_src,c_tgt)
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
                        signal_idt = G(signal_real, c_src, c_src)
                    else:
                        signal_idt = signal_fake
                    if hp.train.rec_loss == 'feat':
                        _, _, features_idt_list = D(signal_idt,c_src,c_src)
                        g_loss_idt = 0
                        for features_idt, features_real in zip(features_idt_list, features_real_list):
                            for feat_idt, feat_real in zip(features_idt, features_real):
                                g_loss_idt += torch.mean(torch.abs(feat_idt - feat_real.detach()))#L1 Loss
                    else:
                        g_loss_idt = torch.mean(torch.abs(signal_real - signal_idt))#L1 Loss
                else:
                    g_loss_idt = 0
                    
                #Latent classification loss
                sig_cont_emb = G.content_embedding
                out_lat_cls = C(sig_cont_emb)
                g_loss_lat_cls = F.cross_entropy(out_lat_cls,label_src)
                lambda_latcls = 1
                
                #Full loss
                g_loss = g_loss_adv_fake + hp.train.lambda_cls*g_loss_cls_fake + hp.train.lambda_rec*g_loss_rec + hp.train.lambda_idt*g_loss_idt + lambda_latcls*g_loss_lat_cls
                #g_loss = g_loss_adv_fake + hp.train.lambda_rec*g_loss_rec + hp.train.lambda_rec*hp.train.lambda_idt*g_loss_idt
                #g_loss = g_loss_adv_fake + hp.train.lambda_cls*g_loss_cls_fake + hp.train.lambda_rec*g_loss_rec + hp.train.lambda_feat*g_loss_feat
                #Optimize
                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                g_loss.backward()
                
                G_grad_norm = torch.norm(torch.stack([param.grad.norm() for param in G.parameters()])).item()
                """
                if G_grad_norm != G_grad_norm:
                    dump = {}
                    dump['signal_fake'] = signal_fake
                    dump['signal_real'] = signal_real
                    dump['grad'] = [param.grad for param in G.parameters()]
                    dump['G_state_dict'] = G.state_dict()
                    dump['D_state_dict'] = D.state_dict()
                    with open(save_path / 'dump','wb') as f:
                        pickle.dump(dump,f)
                    raise Exception('NaN Error')
                """
                
                if hp.train.grad_max_norm_G is not None:
                    torch.nn.utils.clip_grad_norm_(G.parameters(),hp.train.grad_max_norm_G)
                optimizer_G.step()
                
                #Logging
                loss['G_loss_adv_fake'] = g_loss_adv_fake if type(g_loss_adv_fake) == int else g_loss_adv_fake.item() #Check if int because it can be 0
                loss['G_loss_cls_fake'] = g_loss_cls_fake if type(g_loss_cls_fake) == int else g_loss_cls_fake.item()
                loss['G_loss_rec'] = g_loss_rec if type(g_loss_rec) == int else g_loss_rec.item()
                loss['G_loss_idt'] = g_loss_idt if type(g_loss_idt) == int else g_loss_idt.item()
                loss['G_loss_lat_cls'] = g_loss_lat_cls if type(g_loss_lat_cls) == int else g_loss_lat_cls.item()
                
                #G_grad_norm = sum([param.grad.norm().item() for param in G.parameters()])
                G_grad_norm = torch.norm(torch.stack([param.grad.norm() for param in G.parameters()])).item()
                loss['G_loss_grad_norm'] = G_grad_norm
                
                #del out_adv_fake_list, out_cls_fake_list
                #del out_adv_fake, out_cls_fake
                #del g_loss_adv_fake, g_loss_cls_fake
                #del features_rec_list, features_rec, features_real, feat_rec, feat_real, g_loss_rec
                #del g_loss
            
            #Print Losses
            if iter_count % hp.log.log_interval == 0:
                #print(torch.cuda.memory_summary(device=None, abbreviated=False))
                print('Epoch {}/{}, Itt {}'.format(epoch, hp.train.num_epoch, iter_count), end='')
                for label, value in loss.items():
                    logger.add_scalar(label,value,iter_count)
                    print(', {}: {:.4f}'.format(label, value),end='')
                print()
            iter_count += 1
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
                    
                    #Compute fake signal
                    signal_fake = G(signal_real,c_tgt,c_src)
                    #Real signal losses
                    out_adv_real_list, out_cls_real_list, features_real_list = D(signal_real,c_src,c_tgt)
                    #Fake signal losses
                    out_adv_fake_list, out_cls_fake_list, features_fake_list = D(signal_fake.detach(),c_tgt,c_src)
                    
                    d_loss_adv_real = 0
                    d_loss_adv_fake = 0
                    g_loss_adv_fake = 0
                    for out_adv_fake, out_adv_real in zip(out_adv_fake_list,out_adv_real_list):
                        d_loss_adv_real += F.mse_loss(out_adv_real,torch.ones(out_adv_real.size()).to(device))
                        d_loss_adv_fake += F.mse_loss(out_adv_fake,torch.zeros(out_adv_fake.size()).to(device))
                        g_loss_adv_fake += F.mse_loss(out_adv_fake,torch.ones(out_adv_fake.size()).to(device))
                    d_gan_loss = d_loss_adv_real + d_loss_adv_fake
                    
                    d_loss_cls_real = 0
                    d_loss_cls_fake = 0
                    g_loss_cls_fake = 0
                    for out_cls_real,out_cls_fake in zip(out_cls_real_list,out_cls_fake_list):
                        d_loss_cls_real += F.mse_loss(out_cls_real,torch.ones(out_cls_real.size()).to(device))
                        d_loss_cls_fake += F.mse_loss(out_cls_fake,torch.zeros(out_cls_fake.size()).to(device))
                        g_loss_cls_fake += F.mse_loss(out_cls_fake,torch.ones(out_cls_fake.size()).to(device))
                    d_loss_cls = d_loss_cls_real+d_loss_cls_fake
                    
                    sig_cont_emb = G.content_embedding
                    out_lat_cls = C(sig_cont_emb)
                    g_loss_lat_cls = F.cross_entropy(out_lat_cls,label_src)
                    c_acc = torch.sum(torch.argmax(out_lat_cls,dim=1) == label_src)
                    
                    d_loss = d_gan_loss + hp.train.lambda_cls*d_loss_cls
                    g_loss = g_loss_adv_fake + hp.train.lambda_cls*g_loss_cls_fake
                    
                    loss['val_loss_adv_real'] = loss.setdefault('val_loss_adv_real',0) + d_loss_adv_real.item()
                    loss['val_loss_adv_fake'] = loss.setdefault('val_loss_adv_fake',0) + d_loss_adv_fake.item()
                    loss['val_loss_cls_real'] = loss.setdefault('val_loss_cls_real',0) + d_loss_cls_real.item()
                    loss['val_loss_cls_fake'] = loss.setdefault('val_loss_cls_fake',0) + d_loss_cls_fake.item()
                    loss['val_loss_lat_cls'] = loss.setdefault('val_loss_lat_cls',0) + g_loss_lat_cls.item()
                    loss['val_D_loss'] = loss.setdefault('val_D_loss',0) + d_loss.item()
                    loss['val_G_loss'] = loss.setdefault('val_G_loss',0) + g_loss.item()
                    loss['val_C_acc'] = loss.setdefault('val_C_acc',0) + c_acc.item()
                    
            print('Val Epoch {}/{}, Itt {}'.format(epoch, hp.train.num_epoch, iter_count), end='')
            for label, value in loss.items():
                logger.add_scalar(label,value/len(val_data_loader),iter_count)
                print(', {}: {:.4f}'.format(label, value/len(val_data_loader)),end='')
            print()
            G.train()
            D.train()
                    
        #Gen exemples
        if epoch % hp.log.gen_interval == 0:
            print('Saving signals')
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
                
                signal_fake = G(signal_real,c_tgt,c_src)
                signal_rec = G(signal_fake,c_src,c_tgt)
                
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
            with open(save_path / 'latest_epoch','w') as f:
                f.write(str(epoch))
            print('Saved')
                
        #Update random seed
        np.random.seed(initial_seed+epoch)


if __name__ == '__main__':
    main()
