

from datasets import load_dataset, load_metric
from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, LlamaForCausalLM, LlamaTokenizer
import torch
import time
import json
import pandas as pd
import numpy as np
import os
import sys
import random
from peft import PeftModel
import fire
import argparse
import nltk
import re

from inference_t5_prompt_config import *
from  finetune_prepare_datasets_config import *
from  inference_prepare_datasets_config import *
from inference_prepare_datasets_config import *
from chatgpt_infer_config import *
from finetune_llama_prompt_config import *
from inference_prompt_config import *
from finetune_t5_prompt_config import *

from finetune_t5_result_config import *
from finetune_t5_result_base_config import *
from inference_result_t5_config import *
from evaluation_config import *
from finetune_t5_using_llama_test_config import *
from finetune_llama_result_config import *
from inference_result_llama_config import *
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F


print('tag')
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

class DotDict(dict):
    """Dictionary subclass that allows dot notation access to its items."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

global_args = DotDict({

    'multi20_domain_dict' : {'hotel': 0, 'train': 1, 'attraction': 2, 'restaurant': 3, 'taxi': 4}
})



def ensure_directory(path):
    path_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(path_dir):        
        os.makedirs(path_dir)
        print(f"Directory created: {path_dir}")
    else:
        print(f"Directory already exists: {path}, {path_dir}")
    return path


        
def generate_config(config_type, inference_domain, prompt_domain=None, prompt_type = 'test', idx=1, *args, **kwargs):
    if config_type == 'finetune_t5_prompt':
        return gen_finetune_t5_prompt(inference_domain)

    elif config_type == 'inference_t5_prompt':
        return gen_inference_t5_prompt(inference_domain, prompt_domain, prompt_type, idx)

    elif config_type == 'finetune_t5_result':
        return gen_finetune_t5_result(inference_domain)

    elif config_type == 'inference_result_t5':
        return gen_inference_result_t5(inference_domain)
    
    elif config_type == 'evaluation':
        return gen_evaluation(inference_domain)
    



def contrastive_decoding_noncausal(slot_domain, data, model, tokenizer):
    prompt_temp = """
[INST]1.Transform the original slot <Restaurant-Time> into a clearer, user-friendly question that seeks the same information with {}
2.Enrich the new question with the relevant context in {}, like adjectival phrases for the domain or synonyms for the slot [/INST]  """
    prompt_temp_cd = """
[INST]1.Craft a question that incorporates both the domain and the slot found within the dialogue.
2.Enrich the new question with the relevant context in {}, like adjectival phrases for the domain or synonyms for the slot. [/INST] """
    input_seq1 = prompt_temp.format(data['domain_slot_name'], data['dialouge'])  +"\n Optmized Prompt: "
    input_seq2 = prompt_temp_cd.format(data['dialogue'])   +"\n Optmized Prompt: "


    interpolation = 0.9
    input_ids1 = tokenizer(input_seq1, return_tensors='pt', max_length=1024).input_ids.cuda()
    input_ids2 = tokenizer(input_seq2, return_tensors='pt', max_length=1024).input_ids.cuda()

    decoder_start_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(device)

    max_length = 50  
    decoded_sequence = ""

    for _ in range(max_length):
        with torch.no_grad():
            outputs1 = model(input_ids=input_ids1, decoder_input_ids=decoder_input_ids)
            outputs2 = model(input_ids=input_ids2, decoder_input_ids=decoder_input_ids)

            next_token_logits1 = outputs1.logits[:, -1, :]
            next_token_logits2 = outputs2.logits[:, -1, :]
            prob1 = F.log_softmax(next_token_logits1, dim=-1)

            prob2 = F.log_softmax(next_token_logits2, dim=-1)
            mask = torch.zeros_like(prob2, dtype=torch.bool)
            mask[0][prob2.argmax(dim=-1).item()] = 0.01
            prob2 = prob2 * mask

            debiased_prob = prob1 - interpolation * prob2
            next_token_id = debiased_prob.argmax(dim=-1, keepdim=True)
 
        decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)
      
        decoded_sequence = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break
    return decoded_sequence



def get_latest_checkpoint(checkpoint_dirs):
    print(checkpoint_dirs)
    pro_files = os.listdir(checkpoint_dirs)
    latest_steps = 0
    latest_checkpoint = None
    for dir in pro_files:
        try:
            with open(os.path.join(checkpoint_dirs, dir, 'trainer_state.json'), 'r') as f:
                state = json.load(f)
                steps = state.get("global_step", 0)
                if steps > latest_steps:
                    latest_steps = steps
                    latest_checkpoint = dir
        except:
            pass
    return os.path.join(checkpoint_dirs,latest_checkpoint)
print('end load config.py')

