
import os
import sys
import re

'''单个domain'''
def gen_inference_result_t5(inference_domain, *args, **kwargs):
    from config import global_args, DotDict, domain_split, ensure_directory

    inference_type = 'result_finetune'
    batch_size = 16
    micro_batch_size = 4
    num_epochs = 2
    inference_batch_size = 5
    base_model_type = 't5'
    dataset_type = '../../raw_data/multiwoz/data/MultiWOZ_2.1'
    dataset_version = 'MultiWOZ_2.1'
    base_model_path_dict = {}
    base_model_path = base_model_path_dict[base_model_type] 
    auto_batch_size = 8
    auto_micro_batch_size = 4
    auto_num_epochs = 15
    if not os.path.exists(base_model_path):
        print(f"fine tuned mode path {base_model_path} not find!")
        sys.exit(1)   
    assert (
        base_model_path
    ), "Please specify a --model_path, e.g. --model_path='xxx'"
    inference_output_data_dir = ensure_directory(os.path.join(f'{dataset_type}_inference/result/', base_model_type, '_'.join(map(str, ['test', inference_domain, batch_size, micro_batch_size, num_epochs]))))
    lora_weights = os.path.join(f'../../checkpoints/{dataset_version}/', inference_type, base_model_type, '_'.join(map(str, [inference_domain, batch_size, micro_batch_size, num_epochs])))

    
    testfile_list = [ensure_directory(os.path.join(f'../../checkpoints/MultiWOZ_2.1_inference/auto/', base_model_type, '_'.join(map(str, [inference_domain, 'test', batch_size, micro_batch_size, num_epochs]))))]
    base_lora_weights = os.path.join(f'../../checkpoints/{dataset_version}/','result_finetune', base_model_type, '_'.join(map(str, [inference_domain, batch_size, micro_batch_size, num_epochs])))


    inference_result_t5 = DotDict({
        "inference_domain":inference_domain,
        'dataset':'multi20',
        'lora_weights':lora_weights,
        'inference_output_data_dir':inference_output_data_dir,
        'base_model':base_model_path,
        'load_8bit':True,
        'testfile_list':testfile_list,
        'inference_batch_size':inference_batch_size,
        'base_lora_weights':base_lora_weights,
        'max_new_tokens':1024,
        'max_input_length':1024
    })

    return inference_result_t5