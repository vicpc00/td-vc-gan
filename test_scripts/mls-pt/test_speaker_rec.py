import os
import argparse
import pickle
import glob
import re

import numpy as np
import soundfile as sf
import librosa
from fastdtw import fastdtw
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    parser.add_argument('--speechbrain_hparam', required=True)
    args = parser.parse_args()
    return args

#TODO: Load model (pretainer?)
def speechbrain_init(params_file):
    print('aaaa')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin)
        
    #run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected() 
    
    params["embedding_model"].eval()
    
    params["compute_features"].to(device)
    params["mean_var_norm"].to(device)
    params["embedding_model"].to(device)
    params["mean_var_norm_emb"].to(device)
    params["classifier"].to(device)
    
        
    params["label_encoder"].load_or_create(
        path=params["label_encoder_file"]
    )   
       
    print(params["label_encoder_file"],params["label_encoder"].lab2ind)
    
    return params

#TODO: compare embeddings with ref
def speechbrain_speakerrec(test_file, ref_file, speaker_id, sb_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #rs = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
    
    #test_signal,sr = librosa.load(test_file, sr=sb_params['sample_rate'])
    #test_signal = torch.from_numpy(test_signal).unsqueeze(0).to(device)
    test_signal,sr = torchaudio.load(test_file)
    #test_signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(test_signal)
    test_signal = test_signal.to(device)
    #print(test_signal.shape)

    with torch.no_grad():
        speaker_enc = sb_params["label_encoder"].encode_label(speaker_id)
        
        feats = sb_params["compute_features"](test_signal)
        feats = sb_params["mean_var_norm"](feats, torch.tensor([1.0]))
        test_embeddings = sb_params["embedding_model"](feats, torch.tensor([1.0]))

        test_classification = sb_params["classifier"](test_embeddings)
        
        test_embeddings = sb_params["mean_var_norm_emb"](
            test_embeddings, torch.ones(test_embeddings.shape[0]).to(test_embeddings.device)
        )
        
    test_classification = F.softmax(test_classification,dim=2).squeeze()
    test_class = sb_params["label_encoder"].ind2lab[torch.argmax(test_classification).item()]
    #print(torch.argmax(classification), speaker_enc, torch.argmax(classification).item() == speaker_enc)
    #test_correct = torch.argmax(test_classification).item() == speaker_enc
    
    
    """
    #ref_signal,sr = librosa.load(ref_file, sr=sb_params['sample_rate'])
    #ref_signal = torch.from_numpy(ref_signal).unsqueeze(0).to(device)
    ref_signal,sr = torchaudio.load(ref_file)
    ref_signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(ref_signal)
    ref_signal = ref_signal.to(device)
    
    with torch.no_grad():
        speaker_enc = sb_params["label_encoder"].encode_label(speaker_id)
        
        feats = sb_params["compute_features"](ref_signal)
        feats = sb_params["mean_var_norm"](feats, torch.tensor([1.0]))
        ref_embeddings = sb_params["embedding_model"](feats, torch.tensor([1.0]))

        ref_classification = sb_params["classifier"](ref_embeddings)
        
        ref_embeddings = sb_params["mean_var_norm_emb"](
            ref_embeddings, torch.ones(ref_embeddings.shape[0]).to(ref_embeddings.device)
        )
        
    ref_classification = F.softmax(ref_classification,dim=2).squeeze()
    
    ref_class = sb_params["label_encoder"].ind2lab[torch.argmax(ref_classification).item()]
    #ref_correct = torch.argmax(ref_classification).item() == speaker_enc
    
    #emb_dist = torch.norm(ref_embeddings - test_embeddings, dim=2).squeeze().item()
    emb_dist = torch.nn.CosineSimilarity(dim=2)(ref_embeddings,test_embeddings).squeeze().item()
    
    #print(speaker_id, test_class, test_classification[speaker_enc].item(), ref_class, ref_classification[speaker_enc].item(), emb_dist)
    """
    test_embeddings = test_embeddings.squeeze().detach().cpu().numpy()
    #ref_embeddings = ref_embeddings.squeeze().detach().cpu().numpy()
    
    return test_class, test_classification[speaker_enc].item(), test_embeddings
    #return test_class, test_classification[speaker_enc].item(), ref_class, ref_classification[speaker_enc].item(), emb_dist, test_embeddings, ref_embeddings 
        
    
    

def test_speaker_rec():
    args = parse_args()
    
    sb_hparams = speechbrain_init(args.speechbrain_hparam)

    orig_list = glob.glob(os.path.join(args.test_path, '*X_orig.wav'))
    orig_list.sort()
    #print(orig_list)
    
    results = {'test_class':{}, 'test_tgt_prob':{}, 'ref_class':{}, 'ref_tgt_prob':{}, 'emb_dist':{}, 'test_emb':{}, 'ref_emb':{}}

    for src_file in tqdm(orig_list):
        filename, src_spk = re.match('(\S+)_(\S+?)-X_orig.wav',os.path.basename(src_file)).groups()
        
        """
        if src_spk not in dists.keys():
            dists[src_spk] = {}
        """
        
        ref_class, ref_tgt_prob, ref_emb = speechbrain_speakerrec(src_file,'', src_spk, sb_hparams)
        results['ref_class'].setdefault(src_spk,[]).append(ref_class)
        results['ref_tgt_prob'].setdefault(src_spk,[]).append(ref_tgt_prob)
        results['ref_emb'].setdefault(src_spk,[]).append(ref_emb)
        
        #print(filename,src_spk)
        conv_list =  glob.glob(os.path.join(args.test_path, f'{filename}_{src_spk}-*_conv.wav'))
        #print(conv_list)
        for conv_file in conv_list:
            #print(f'{filename}_{src_spk}-(\S+?)_conv.wav',conv_file)
            tgt_spk = re.match(f'{filename}_{src_spk}-(\S+?)_conv.wav',os.path.basename(conv_file)).group(1)
            """
            if tgt_spk not in dists[src_spk].keys():
                dists[src_spk][tgt_spk] = []
            """
            tgt_file = os.path.join(args.test_path,f'{re.sub(src_spk,tgt_spk,filename)}_{tgt_spk}-X_orig.wav')
            #print(src_file, tgt_file, conv_file)
            
            #test_class, test_tgt_prob, ref_class, ref_tgt_prob, emb_dist, test_emb, ref_emb = speechbrain_speakerrec(conv_file,tgt_file, tgt_spk, sb_hparams)
            test_class, test_tgt_prob, test_emb = speechbrain_speakerrec(conv_file,tgt_file, tgt_spk, sb_hparams)
            
            results['test_class'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(test_class)
            results['test_tgt_prob'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(test_tgt_prob)
            #results['emb_dist'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(emb_dist)
            results['test_emb'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(test_emb)
            

    spks = results['ref_class'].keys()
    for tgt_spk in spks:
        
        mean_emb = np.mean(results['ref_emb'][tgt_spk],axis=0)
        
        for src_spk in spks:
            results['emb_dist'].setdefault(src_spk,{})[tgt_spk] = [
                    torch.nn.CosineSimilarity(dim=0)(torch.tensor(mean_emb),torch.tensor(test_emb)).squeeze().item()
                    for test_emb in results['test_emb'][src_spk][tgt_spk]]
            
            
            
    with open(args.save_file,'wb') as f:
        pickle.dump(results,f)
    
    """
    s = 0
    n = 0
    for src_spk in results['test_class'].keys():
        for tgt_spk in results['test_class'].keys():
            print(src_spk, tgt_spk, results['test_class'][src_spk][tgt_spk])
            print(src_spk, tgt_spk, results['ref_class'][src_spk][tgt_spk])
            s += sum([spk==tgt_spk for spk in results['test_class'][src_spk][tgt_spk]])
            n += len(results['test_class'][src_spk][tgt_spk])
    print(f'Correct rate = {s/n}')
    """

if __name__ == '__main__':
    test_speaker_rec()