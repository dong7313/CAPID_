
import os
import sys
import re


def gen_finetune_t5_prompt(inference_domain, *args, **kwargs):
    from config import global_args,DotDict
    multi20_domain = list(global_args.multi20_domain_dict.keys())
    multi20_domain.remove(inference_domain)


    batch_size = 8
    micro_batch_size = 4
    num_epochs = 15
    max_input_length = 1024
    max_target_length = 128
    # t5_base: "your t5_base model path"
    base_model_path_dict = {'t5_base':'google-t5/t5-base'}

    base_model_type = 't5_base'
    base_model_path = base_model_path_dict[base_model_type] 
    # if not os.path.exists(base_model_path):
    #     print(f"fine tuned mode path {base_model_path} not find!")
    #     sys.exit(1)   
    # assert (
    #     base_model_path
    # ), "Please specify a --model_path, e.g. --model_path='xxx'"

    # output_dir
    base_model_output_dir = os.path.join('../../checkpoints/MultiWOZ_2.1/', base_model_type, '_'.join(map(str, [inference_domain, batch_size, micro_batch_size, num_epochs])))

    # data_path
    data_path_list = []


    chatgpt_data_dir = f'../../raw_data/multiwoz/data/MultiWOZ_2.1_chatgpt/base/'
    for domain in multi20_domain:
        if os.path.isfile(os.path.join(chatgpt_data_dir, f'test_{domain}_chat_LLM_zero-shot.json')):
            data_path_list.append(os.path.join(chatgpt_data_dir, f'test_{domain}_chat_LLM_zero-shot.json'))

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