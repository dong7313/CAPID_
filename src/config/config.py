

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
    


def calibrate_prompt_cd(slot_domain, answer, instruction, model = model_t5_prompt, tokenizer = tokenizer_t5_prompt):
    domain = slot_domain.split('_')[0]
    slot = slot_domain.split('_')[1]
    if domain not in answer:
        calibrate_prompt_cd = contrastive_decoding_noncausal(slot_domain, instruction, model, tokenizer)
    else:
        return answer




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

