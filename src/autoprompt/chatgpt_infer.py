import requests
import json
import os
import time
import random
import sys
import fire
from collections import defaultdict
sys.path.append('../config')
from config import *


def chat_gpt(args, messages, counter, error_count, s_time):
    responses = []
    for i, m in enumerate(messages):
        try:
            message = m['message']

            data = json.dumps({"model": "gpt-4", "messages": message, 'temperature':0.9})
            print(message)
            response = requests.post(args.API_URL, headers=args.HEADERS, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']

            m['response'] = res
            # save to file
            with open(args.chatgpt_output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(response_json)
            counter += 1
        except Exception as e:
            error_count += 1
           
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter, error_count), end='\r')

    return responses


def get_messages_list(args):
    data = json.load(open(args.chatgpt_input_file))

    messages_list = []
    slot_type = defaultdict(float)

    data_type = defaultdict(float)
    for idx, i in enumerate(data):
        slot_type[i['domain_slot_name'].split('_')[1]] += 1

        if i['groundtruth'] in ['None', 'NONE', 'not mentioned']:    
            data_type['none'] += 1
        else:
            data_type['other'] += 1
    all_sample = 10000.0
    print(f'data_type:{data_type}')
    print(all_sample / data_type['none'],  all_sample / data_type['other'])




    for idx, i in enumerate(data):

        if i['groundtruth'] == i['model_output'] or i['groundtruth'][1:] == i['model_output']: 
            continue
        slot = i['domain_slot_name'].split('_')[1]

        if i['groundtruth'] in ['None', 'NONE', 'not mentioned']:    
            if random.random() > all_sample / data_type['none']:
                continue
        else:
            if random.random() > all_sample / data_type['other']:
                continue
               
        text = args.ctx_prompt.format(slot = i['domain_slot_name'], dialgoue = i['dialogue'], groundtruth = i['groundtruth'])

        messages_list.append({
            'message': [
                {"role": "user", "content": text}
            ],
            'origin': i
        })
    return messages_list



def chatgpt_infer(inference_domain, config_type = 'chatgpt_infer'):

    print('Start now')
    args = generate_config(config_type, inference_domain)

    messages_list = get_messages_list(args)
    print("Total num: ", len(messages_list))

    # s_time = time.time()
    # responses = chat_gpt(args, messages_list, 0, 0, s_time)




if __name__ == '__main__':
    fire.Fire(chatgpt_infer)