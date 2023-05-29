__package__ = "common"

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

import resemblyzer

from . import parse_fn as default_parse_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    args = parser.parse_args()
    return args


def speechbrain_spkrec_compare(test_file, ref_file, speaker_id, sb_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #rs = torchaudio.transforms.Resample(orig_freq=16000, new_freq=48000)
    
    #test_signal,sr = librosa.load(test_file, sr=sb_params['sample_rate'])
    #test_signal = torch.from_numpy(test_signal).unsqueeze(0).to(device)
    test_signal,sr = torchaudio.load(test_file)
    test_signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(test_signal)
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
    
    test_embeddings = test_embeddings.squeeze().detach().cpu().numpy()
    ref_embeddings = ref_embeddings.squeeze().detach().cpu().numpy()

    return test_class, test_classification[speaker_enc].item(), ref_class, ref_classification[speaker_enc].item(), emb_dist, test_embeddings, ref_embeddings 
        
def speechbrain_spkrec_single(test_file, speaker_id, sb_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_signal,sr = torchaudio.load(test_file)
    test_signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)(test_signal)
    test_signal = test_signal.to(device)
    
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
        
    test_embeddings = test_embeddings.squeeze().detach().cpu().numpy()
    return test_class, test_classification[speaker_enc].item(), test_embeddings

def resemblyzer_spkrec_single(test_file, speaker_id, encoder):
    test_signal,sr = librosa.load(test_file, sr=16000)
    
    test_signal = resemblyzer.preprocess_wav(test_signal, source_sr = sr)
    test_embeddings = encoder.embed_utterance(test_signal)
    
        
    test_class = 'p000'
        
    return test_class, test_embeddings


def test_speaker_rec(out_filename, test_dir, parse_fn = None):
    if not parse_fn:
        parse_fn = default_parse_fn
    
    orig_list = glob.glob(os.path.join(test_dir, '*X-orig.wav'))
    orig_list.sort()
    #print(orig_list)
    
    encoder = resemblyzer.VoiceEncoder()
    
    results = {'test_class':{}, 'test_tgt_prob':{}, 'ref_class':{}, 'ref_tgt_prob':{}, 'emb_dist':{}, 'test_emb':{}, 'ref_emb':{}}

    for src_file in tqdm(orig_list):
        sig_id, src_spk, _, _ = parse_fn(src_file)
        
        ref_class, ref_emb = resemblyzer_spkrec_single(src_file, src_spk, encoder)
        results['ref_class'].setdefault(src_spk,[]).append(ref_class)
        #results['ref_tgt_prob'].setdefault(src_spk,[]).append(ref_tgt_prob)
        results['ref_emb'].setdefault(src_spk,[]).append(ref_emb)
        
        #print(filename,src_spk)
        conv_list =  glob.glob(os.path.join(test_dir, f'{sig_id}-{src_spk}-*-conv.wav'))
        #print(conv_list)
        for conv_file in conv_list:
            _, _, tgt_spk, _ = parse_fn(conv_file)
            
            test_class, test_emb = resemblyzer_spkrec_single(conv_file, tgt_spk, encoder)
            
            results['test_emb'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(test_emb)
            
            
    spks = list(results['ref_class'].keys())
    mean_emb = []
    for tgt_spk in spks:
        
        mean_emb.append(np.mean(results['ref_emb'][tgt_spk],axis=0))
        
        for src_spk in spks:
            results['emb_dist'].setdefault(src_spk,{})[tgt_spk] = [
                    torch.nn.CosineSimilarity(dim=0)(torch.tensor(mean_emb[-1]),torch.tensor(test_emb)).squeeze().item()
                    for test_emb in results['test_emb'][src_spk][tgt_spk]]
    for src_spk in spks:
        results['test_class'].setdefault(src_spk,{})
        for tgt_spk in spks:
            dists = np.linalg.norm(np.array(mean_emb)[np.newaxis,:,:] - np.array(results['test_emb'][src_spk][tgt_spk])[:,np.newaxis,:], axis=2)
            #print(dists.shape)
            idxs = np.argmin(dists,axis=1)
            #print(idxs.shape)
            results['test_class'][src_spk][tgt_spk] = [spks[idx] for idx in idxs]
            
    with open(out_filename,'wb') as f:
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
    args = parse_args()
    test_speaker_rec(args.save_file, args.test_path)
