import os
import sys
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import torch
import time
#import evaluate
import pandas as pd
import numpy as np
import os
import sys
from peft import PeftModel
import fire
import json

import argparse
import nltk
from transformers import set_seed
# import config
from config import *









def finetune_t5_prompt_bak(inference_domain,  config_type = 'finetune_t5_prompt'):

    args = generate_config(config_type,inference_domain)



    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(args)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    padding = False #"max_length"
    ignore_pad_token_for_loss = args.ignore_pad_token_for_loss

    prefix = ""
    def preprocess_function(examples):

        prompt_temp = """
[INST]You are an expert prompt engineer.
1.Transform the original slot <Restaurant-Time> into a clearer, user-friendly question that seeks the same information with <Restaurant-Time>.
2.Enrich the new question with the relevant context in [Dialogue], like adjectival phrases for the domain or synonyms for the slot[/INST] \
 "\nOriginal Slot:" """
        inputs = [prompt_temp + examples['domain_slot_name'][idx] +"\nOriginal Prompt: " + examples['instruction'][idx]  +"\n Optimized Prompt: " for idx in range(len(examples['instruction'])) ]
        targets = examples['optimized_instruction']
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)

        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    

    # load datasets

    for path in args.data_path_list:
        assert path.endswith('.json') or path.endswith('.jsonl'), "All files must be in .json or .jsonl format"



    cur_data_path_list = []
    for data_path in args.data_path_list:
        result_data = []
        if 'otherthree' not in data_path:
            continue
        for i in open(data_path).readlines():
            i = json.loads(i)
            res = {'instruction':i['origin']['instruction'], 
                'dialogue': i['origin']['instruction'],
                'Adjectival phrase':i['response'].split('[END]')[0].replace('Adjectival phrase:', ''),
                'optimized_instruction':i['response'].split('[END]')[1].replace('\nOptimized Instruction:', ''),
                'domain_slot_name':i['origin']['domain_slot_name']
                }
            if res['domain_slot_name'].split('_')[0] not in res['optimized_instruction']:
                continue

            result_data.append(res)
        print(data_path.replace('LLM_zero-shot.json', 'finetune'))

        with open(data_path.replace('LLM_zero-shot.json', 'finetune'),  'w') as f:
            json.dump(result_data, f, indent=4)
        cur_data_path_list.append(data_path.replace('LLM_zero-shot.json', 'finetune'))
    raw_datasets = load_dataset("json", data_files=cur_data_path_list)



        
    val_set_size = 100
    if val_set_size > 0:
        train_val = raw_datasets["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(preprocess_function, batched=True,batch_size=500)
        )
        val_data = (
            train_val["test"].shuffle().map(preprocess_function, batched=True,batch_size=500)
        )
    else:
        train_data = raw_datasets["train"].shuffle().map(preprocess_function, batched=True)
        val_data = None
    print(f"train_data: {train_data}")
    print(f"val_data: {val_data}")    

    
    metric = load_metric("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    tokenized_datasets = raw_datasets.shuffle().map(preprocess_function, batched=True,desc="Running tokenizer on train dataset")


    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    batch_size = args.batch_size
    model_name = args.model_path.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy = "steps",
        save_strategy="steps",
        learning_rate = 3e-4,
        warmup_steps=50,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        weight_decay = 0.01,
        save_total_limit =2,
        load_best_model_at_end=True,
        eval_steps=500,
        save_steps=500,
        output_dir=args.output_dir,
        num_train_epochs = args.num_epochs,
        predict_with_generate = True,
        fp16 = True,
        push_to_hub = False,
        #logging_dir=log_dir,
    )
    
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )
    
    
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset = train_data,
        eval_dataset = val_data,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    
    #resume_from_checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output_dir)
    print(f'already saved at{args.output_dir} ')
  

if __name__ == "__main__":

    fire.Fire(finetune_t5_prompt_bak)