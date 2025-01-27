

import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import time

from config import *

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



def inference_t5_prompt(inference_domain, prompt_domain, prompt_type, idx, gpu=0, config_type = 'inference_t5_prompt'):
    args = generate_config(config_type,inference_domain, prompt_domain, prompt_type, idx)


    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.lora_weights)
    
    model.to(device)
    model.eval()

    # load inference data
    data = json.load(open(args.testfile_list[0]))
    evaluated = set()
    print(args.inference_output_data_dir)
    if os.path.exists(args.inference_output_data_dir):
        for line in open(args.inference_output_data_dir).readlines():
            try:
                line = json.loads(line)
                evaluated.add(line['idx']+line['domain_slot_name'])
            except:
                pass
    s_time = time.time()

    for idx_ in range(len(data)):
        sample = data[idx_]
        if sample['idx'] + sample['domain_slot_name'] in evaluated:
            continue
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        sample['optimized_output'] = contrastive_decoding_noncausal(sample, model, tokenizer)

        with open(args.inference_output_data_dir, 'a', encoding='utf-8') as f:
            print(json.dumps(sample, ensure_ascii=False), file=f)
        print(f'running time:{time.time()-s_time} idx: {idx_}, len: {len(data)}')


if __name__ == "__main__":
    fire.Fire(inference_t5_prompt)