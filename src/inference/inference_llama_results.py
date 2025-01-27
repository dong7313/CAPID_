# 生成计算averageJGA指标所需要的预测结果

import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
import random
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
sys.path.append('/data_8T2/xiaoyu/exp_test/auDST/src/config')
from config import  ensure_directory
import time
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass



# dataset_type = '../../raw_data/multiwoz/data/MultiWOZ_2.1'
dataset_type = '../../raw_data/MultiWOZ2.4/data/mwz24'
# dataset_version = 'MultiWOZ_2.1'
dataset_version = 'MultiWOZ_2.4'



def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    exp_type: str = 'exp', # exp or qa 'Question-answering' 'Schema-driven Prompting'
    base_model_type: str = "llama2",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    testfile_name: str = "",
    testfile_idx: str = "",
    cutoff_len:int = 1024,
     sample: str = '16000',
    output_file: str = "",
    lora_type: str = "-averaging",
    dataset_id: int = 1, # 1 - 5  5次实验
    inference_domain: str = "",
    split_id:int = 0,
):


    batch_size = 128
    micro_batch_size = 4
    num_epochs = 1
    lora_weights = os.path.join(f'../../checkpoints/{dataset_version}/','result_finetune', f'llama2_{exp_type}_{sample}', '_'.join(map(str, [inference_domain, batch_size, micro_batch_size, num_epochs])))


    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)   
    assert (
        lora_weights
    ), "Please specify a --lora_weights, e.g. --lora_weights='xxx'"

    
    output_dir = ensure_directory(os.path.join(f'{dataset_type}_inference/result/', f'llama2_{exp_type}_{sample}', '_'.join(map(str, [inference_domain, 'test', batch_size, micro_batch_size, num_epochs]))))


    print(f"lora_weights: {lora_weights}")
    print(f"output_dir: {output_dir}")
    
    with open('../../raw_data/multiwoz/data/MultiWOZ_2.1/slot_descriptions.json', 'r') as file:
        slot_descripstions_file = json.load(file)
    slot_descripstions = {}
    for slot, desp in slot_descripstions_file.items():
        d, s = slot.split('-')
        slot_descripstions['_'.join([d.lower(), s.strip('book ').lower()])] = desp[1] 



    tokenizer = LlamaTokenizer.from_pretrained(base_model)


    
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit, 
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        data_point,
        input=None,
        temperature=0.02,
        top_p=0,
        top_k=1,
        num_beams=1,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):

        if data_point['possible_values'] != '' and len(data_point['possible_values']) < 300:
            full_prompt = data_point['optimized_output_cd_cal'] + 'You can only answer from the following available values: None, ' + data_point['possible_values'] + '\n [DIALOGUE_CONTEXT]:' + data_point['dialogue'] + ' \n Output:' 
        else:
            full_prompt = data_point['optimized_output_cd_cal'] + 'If the information is not mentioned, just return None. ' + '\n [DIALOGUE_CONTEXT]:' + data_point['dialogue'] + ' \n Output:' 
        

        inputs = tokenizer(full_prompt, max_length=cutoff_len,return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output, full_prompt



    testfile_name = os.path.join(f'{dataset_type}_inference/auto/', 't5_base_cd', '_'.join(map(str, ['hotel', 'train', inference_domain, 16, 4, 10])) + '.json')

    
    print(f"test filename: {testfile_name}")
    s_time = time.time()


    data = json.load(open(testfile_name))
    split_size = int(len(data) / 4)
    print(f'ssplit_size {split_size}')


    data = data[split_id * split_size:(split_id + 1) * split_size]
    print(f'lendata{len(data)}, {split_id * split_size},{(split_id + 1) * split_size}')

    evaluated = set()

    if os.path.exists(output_dir):
        for line in open(output_dir).readlines():
            try:
                line = json.loads(line)
                evaluated.add(line['idx']+line['domain_slot_name'])
            except:
                pass


    for idx_ in range(len(data)):
        sample = data[idx_]
        if sample['idx']+sample['domain_slot_name'] in evaluated:
            continue
        Response, sample['instruction']= evaluate(sample)
        res = Response.split('Output:')[1].split('</s>')[0].strip('\n')


        sample['optimized_model_output'] = res
        with open(output_dir, 'a', encoding='utf-8') as f:
            print(json.dumps(sample, ensure_ascii=False), file=f)


        print(f'running time:{time.time()-s_time} idx: {idx_}, len: {len(data)}')

if __name__ == "__main__":
    fire.Fire(main)
