__package__ = "common"

import os
import argparse
import pickle
import glob
import re

import numpy as np
import soundfile as sf
import librosa
import pysptk
from tqdm import tqdm

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate 

from . import parse_fn as default_parse_fn
# def default_parse_fn(filename):
#     phrase_id, src_spk, tgt_spk, sig_type = re.match(r'(\d+)-(\S+)-(\S+)-(orig|conv).wav',os.path.basename(filename)).groups()
#     return phrase_id, src_spk, tgt_spk, sig_type
# def name_fn(spk):
#     return spk.split('_')[-1]

import warnings
#warnings.filterwarnings("error")

ref_trans = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def transcribe(test_file, ref_file, model, processor):
    
    
    test_signal,sr = librosa.load(test_file, sr=16000)
    
    input_features = processor(test_signal, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to(device))[0]
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    transcription = processor.tokenizer._normalize(transcription)
    
    if ref_file in ref_trans:
        reference = ref_trans[ref_file]
    else:
        with open(ref_file, 'r') as f:
            reference = processor.tokenizer._normalize(f.read())
        
    return transcription, reference



def test_asr(out_filename, test_dir, transc_dir, parse_fn = None, name_fn = None, language = "portuguese"):
    if not parse_fn:
        parse_fn = default_parse_fn
    if not name_fn:
        name_fn = lambda name: name
        
    model_name = "openai/whisper-medium"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")
    
    orig_list = glob.glob(os.path.join(test_dir, '*X-orig.wav'))
    orig_list.sort()
    #print(orig_list)
    
    results={'asr_results_transcs':{},'asr_results_refs':{}, 
             'asr_results_transcs_orig':{},'asr_results_refs_orig':{}, 
             'asr_results_wer_pair':{}, 'asr_results_cer_pair':{}, 
             'asr_results_wer':{}, 'asr_results_cer':{}, 
             'asr_results_orig_wer_single':{}, 'asr_results_orig_cer_single':{}, 
             'asr_results_orig_wer':{}, 'asr_results_orig_cer':{}}
    
    for src_file in tqdm(orig_list):
        sig_id, src_spk, _, _ = parse_fn(src_file)
        ref_file = os.path.join(transc_dir, f'{name_fn(src_spk)}-{sig_id}.txt')

        conv_list =  glob.glob(os.path.join(test_dir, f'{sig_id}-{src_spk}-*-conv.wav'))
        for conv_file in conv_list:
            _, _, tgt_spk, _ = parse_fn(conv_file)
           
            transcription, reference = transcribe(conv_file, ref_file, model, processor)
            results['asr_results_transcs'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(transcription)
            results['asr_results_refs'].setdefault(src_spk,{}).setdefault(tgt_spk,[]).append(reference)
            
        transcription, reference = transcribe(src_file, ref_file, model, processor)
        results['asr_results_transcs_orig'].setdefault(src_spk,[]).append(transcription)
        results['asr_results_refs_orig'].setdefault(src_spk,[]).append(reference)
            
        
    wer_eval = evaluate.load('wer')
    cer_eval = evaluate.load('cer')
    spks = list(results['asr_results_transcs'].keys())
    
    all_transcs = []
    all_refs = []
    all_transcs_orig = []
    all_refs_orig = []
    for src_spk in spks:
        for tgt_spk in spks:
            transcs = results['asr_results_transcs'][src_spk][tgt_spk]
            refs = results['asr_results_transcs_orig'][src_spk]
            results['asr_results_wer_pair'].setdefault(src_spk,{})[tgt_spk] = wer_eval.compute(predictions=transcs, references=refs)
            results['asr_results_cer_pair'].setdefault(src_spk,{})[tgt_spk] = cer_eval.compute(predictions=transcs, references=refs)
            all_transcs += transcs
            all_refs += refs
        transcs = results['asr_results_transcs_orig'][src_spk]
        refs = results['asr_results_refs_orig'][src_spk]
        results['asr_results_orig_wer_single'][src_spk] = wer_eval.compute(predictions=transcs, references=refs)
        results['asr_results_orig_cer_single'][src_spk] = cer_eval.compute(predictions=transcs, references=refs)
        all_transcs_orig += transcs
        all_refs_orig += refs
        
    results['asr_results_wer'] = wer_eval.compute(predictions=all_transcs, references=all_refs)
    results['asr_results_cer'] = cer_eval.compute(predictions=all_transcs, references=all_refs)
    results['asr_results_orig_wer'] = wer_eval.compute(predictions=all_transcs_orig, references=all_refs_orig)
    results['asr_results_orig_cer'] = cer_eval.compute(predictions=all_transcs_orig, references=all_refs_orig)
            
    
    with open(out_filename,'wb') as f:
        pickle.dump(results,f)

    # for src_spk in spks:
    #     for tgt_spk in spks:
    #         print(src_spk, tgt_spk, results['asr_results_wer_pair'][src_spk][tgt_spk], results['asr_results_cer_pair'][src_spk][tgt_spk])
    # print(results['asr_results_wer'], results['asr_results_cer'])
    
    # for src_spk in spks:
    #     print(src_spk, results['asr_results_orig_wer_single'][src_spk], results['asr_results_orig_cer_single'][src_spk])
    # print(results['asr_results_orig_wer'], results['asr_results_orig_cer'])
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--save_file', required=True)
    parser.add_argument('--transc_dir', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    test_asr(args.save_file, args.test_path, args.transc_dir)

