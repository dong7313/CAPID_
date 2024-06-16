
import os
import sys
import re

'''单个domain'''
def gen_finetune_t5_prompt(inference_domain):
    from config import global_args,DotDict,domain_split
    train_domain = domain_split[inference_domain][0]
    chatgpt_domain = domain_split[inference_domain][1]
    multi20_domain = list(global_args.multi20_domain_dict.keys())
    multi20_domain.remove(inference_domain)

    #一些需要继续调整的参数
    batch_size = 8
    micro_batch_size = 4
    num_epochs = 15

    max_input_length = 1024
    max_target_length = 128

    base_model_path_dict = {}
    base_model_type = 't5_base'
    base_model_path = base_model_path_dict[base_model_type] 
    if not os.path.exists(base_model_path):
        print(f"fine tuned mode path {base_model_path} not find!")
        sys.exit(1)   
    assert (
        base_model_path
    ), "Please specify a --model_path, e.g. --model_path='xxx'"


    # output_dir
    base_model_output_dir = os.path.join('../../checkpoints/MultiWOZ_2.1/', base_model_type, '_'.join(map(str, [inference_domain, batch_size, micro_batch_size, num_epochs])))

    # data_path
    data_path_list = []

    chatgpt_data_dir = '../../raw_data/multiwoz/data/MultiWOZ_2.1_chatgpt/base_bak'
    for filename in os.listdir(chatgpt_data_dir):
        if os.path.isfile(os.path.join(chatgpt_data_dir, filename)) and 'json' in filename:
            data_path_list.append(os.path.join(chatgpt_data_dir, filename))
  
    val_data_path_list = None

    #gradient_accumulation_steps
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size



    gen_finetune_t5_prompt = DotDict({
        "inference_domain":inference_domain,
        'dataset':'multi20',
        'finetune_domain':multi20_domain,
        'base_model_type':base_model_type,
        'model_path': base_model_path,
        'device_map':device_map,
        'base_model': base_model_path,  # the only required argument
        'output_dir': base_model_output_dir,
    'batch_size':batch_size,
    'num_epochs': num_epochs,
        'ignore_pad_token_for_loss': True,
        'num_epochs': num_epochs,
        'max_input_length':max_input_length,
        'max_target_length':max_target_length,
        'output_dir': base_model_output_dir,
        'data_path_list': data_path_list,
        'val_data_path_list':val_data_path_list,
        'resume_from_checkpoint':None,
        'train_on_inputs':  False,  # if False, masks out inputs in loss
        'add_eos_token':  True,


    })

    return gen_finetune_t5_prompt