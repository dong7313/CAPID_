

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
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import *
import pandas as pd
import random

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def inference_result_t5(inference_domain, config_type = 'inference_result_t5'):
    args = generate_config(config_type, inference_domain)


    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.lora_weights)
    model_flag = False
    try:
        model_base = AutoModelForSeq2SeqLM.from_pretrained(args.base_lora_weights)
        model_base.to(device)
        model_base.eval()
        model_flag = True
    except:
        pass
    model.to(device)
    model.eval()
    print(f'model flag {model_flag},       {args.base_lora_weights}')
    result_out = []
    data = json.load(open(args.testfile_list[0]))
    s_time = time.time()
    for idx_ in range(len(data)):
        sample = data[idx_]
        if sample['possible_values'] != '':
            input_ids = tokenizer(sample['optimized_output_cd_cal']+ 'You can only answer from the following available values: None, ' + sample['possible_values'] + '\n [DIALOGUE_CONTEXT]:' + sample['dialogue'],  return_tensors='pt').input_ids.cuda()
        else:
            input_ids = tokenizer(sample['optimized_output_cd_cal']+ 'If the information is not mentioned, just return None. '+ '\n [DIALOGUE_CONTEXT]:' + sample['dialogue'],  return_tensors='pt').input_ids.cuda()
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        answer = tokenizer.decode(output[0])
        if "</s>" in answer:
            answer = answer.replace("</s>","")
        if "<pad>" in answer:
            answer = answer.replace("<pad>","")
        sample['optimized_model_output'] = answer.strip(' ')
        result_out.append(sample)

        print(f'running time:{time.time()-s_time} idx: {idx_}, len: {len(data)}')

    with open(args.inference_output_data_dir, 'w') as f:
        json.dump(result_out, f, indent=4)




if __name__ == "__main__":
    fire.Fire(inference_result_t5)  
