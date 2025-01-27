import requests
import json
import os
import sys
import time
import random
import fire

sys.path.append('../finetune')
sys.path.append('../utils')
sys.path.append('../inference')
sys.path.append('../config')
sys.path.append('/home/data2/xiaoyu/exp_test/CAPID/src/config')
from config import *

from chatgpt_infer import *



def autoprompt(inference_domain, config_type = 'chatgpt_infer'):

    print('start now')
    args = generate_config(config_type, inference_domain)


    messages_list = get_messages_list(args)
    print("total num: ", len(messages_list))

    s_time = time.time()
    responses = chat_gpt(args, messages_list, 0, 0, s_time)


if __name__ == '__main__':
    fire.Fire(autoprompt)

