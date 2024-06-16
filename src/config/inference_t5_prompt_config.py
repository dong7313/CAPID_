
import os
import sys
import re

def gen_inference_t5_prompt(inference_domain, prompt_domain, prompt_type, idx):
    from config import global_args, DotDict, domain_split, ensure_directory
    print(inference_domain, prompt_domain, prompt_type)
    train_domain = domain_split[inference_domain][0]
    chatgpt_domain = domain_split[inference_domain][1]
    multi20_domain = list(global_args.multi20_domain_dict.keys())
    multi20_domain.remove(inference_domain)

    #一些需要继续调整的参数
    batch_size = 16
    micro_batch_size = 4
    num_epochs = 10
    inference_batch_size = 5
    base_model_type = 't5_base_cd'
    if not os.path.exists(base_model_path):
        print(f"fine tuned mode path {base_model_path} not find!")
        sys.exit(1)   
    assert (
        base_model_path
    ), "Please specify a --model_path, e.g. --model_path='xxx'"
    lora_weights = ensure_directory(os.path.join('../../checkpoints/MultiWOZ_2.1/', base_model_type, '_'.join(map(str, [inference_domain, batch_size, micro_batch_size, num_epochs]))))
    lora_weights = ensure_directory('../..//checkpoints/MultiWOZ_2.1/t5_base/taxi_8_4_15')
    inference_output_data_dir = ensure_directory(os.path.join(f'{dataset_type}_inference/auto/', base_model_type, '_'.join(map(str, [inference_domain, prompt_type, prompt_domain, batch_size, micro_batch_size, num_epochs]))))
    testfile_list = [ensure_directory(f'{dataset_type}_optimized_full/{prompt_type}_{prompt_domain}_{idx}_LLM_zero-shot.json')]
    inference_t5_prompt = DotDict({
        "inference_domain":inference_domain,
        'dataset':'multi20',
        'lora_weights':lora_weights,
        'inference_output_data_dir':inference_output_data_dir,
        'base_model':base_model_path,
        'load_8bit':True,
        'testfile_list':testfile_list,
        'inference_batch_size':inference_batch_size,
        'max_new_tokens':512,
        'base_model_path':base_model_path,



    })

    return inference_t5_prompt