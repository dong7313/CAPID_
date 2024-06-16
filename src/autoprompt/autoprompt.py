import requests
import json
import os
import sys
import time
import random
import fire

from config import *

from autoprompt_utils import * 
from finetune import *
from inference import *
from chatgpt_infer import *

def main():
    fire.Fire(chatgpt_infer(chatgpt_info_args))
    '''Step3: teach student model'''
    finetune_info_args['data_path_list'] = chatgpt_info_args.chatgpt_output_file 
    finetune_info_args['model_path'] = '/home/yujie/llama_weights_hf'
    finetune_info_args['output_dir'] = os.path.join('../../checkpoints/auto_model/', finetune_info_args.base_model_type, finetune_info_args.inference_domain)
    fire.Fire(finetune(finetune_info_args))
    '''Step4: inference on student model      revise input output inference_key'''
    fire.Fire(inference(generate_inference_info_args('student')))
    '''Step5: inference on base model'''
    fire.Fire(inference(generate_inference_info_args('auto')))



if __name__ == '__main__':
    main()

