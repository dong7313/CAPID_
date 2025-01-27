
# instruction + input + output

# zero-shot experiment



import random
import json
import sys
from collections import defaultdict


def Template1(dial_text):
    dial = dial_text['dialogue'].split(" [domain] ")
    input1 = ""
    input1 = input1 + dial[0] + " \n "
    possible_values =''
    slot_possible= ''

    if "Possible Values" in dial[1]:
        dial2 = dial[1].split(" [Possible Values] ")
        slot_possible = dial2[1]
        
        input1 = input1 + "[domain] " + dial2[0] + " "
        if random.random() > 0.5:
            input1 = input1 + "This slot is categorical and you can only choose from the following available values: "
            input1 = input1 + dial2[1] + ". "
            possible_values = "This slot is categorical and you can only choose from the following available values: " +  dial2[1] + ". "
    else:
        input1 = input1 + "[domain] " + dial[1] + " "
    input1 = input1 + "If the slot is not mentioned in the dialogue, just return NONE. \n "
    possible_values += "If the slot is not mentioned in the dialogue, just return NONE. \n "
    
    if random.random() > 0.5:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is \n"
    else:
        input1 = input1 + "So the value of slot <"+ d_name+ "-" + s_name +"> is "
    
    output1 = dial_text['state']
    return input1, output1, possible_values, slot_possible



if __name__ == '__main__':
    frame_idxs = {"train": 0, "taxi":1, "bus":2, "police":3, "hotel":4, "restaurant":5, "attraction":6, "hospital":7}
 
    except_domain = "attraction" # delete this domain from traning set
    except_number = 0
    original_number = 0
    for data_type in ["train", 'test', 'dev']:   
        data_dir = "../../raw_data/multiwoz/data/MultiWOZ_2.1_preprocess/"+ data_type +".json"
        data_idx = "../../raw_data/multiwoz/data/MultiWOZ_2.1_preprocess/"+ data_type +".idx"
        output_filename = "../../raw_data/multiwoz/data/MultiWOZ_2.1_preprocess/"+ data_type +"_LLM_zero-shot_except-"+ str(except_domain) +"-domain.json"
        dataset_data = defaultdict(list)
        idx_lines = open(data_idx).readlines()
        test_data_lines = open(data_dir).readlines()
        assert len(idx_lines) == len(test_data_lines)
        #sys.exit(1)
        for idx_ in range(len(idx_lines)):
            dial_text = eval(test_data_lines[idx_].strip())
            original_number += 1
            item = {}
            idx_list = idx_lines[idx_].strip()
            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_list.split("|||") 
            dic_key_name = '_'.join([dial_idx, turn_idx])
            domain_slot_name = '_'.join([d_name, s_name])

            input1, output1, possible_values, slot_possible = Template1(dial_text)
            item['dialogue'] = dial_text['dialogue'].split(" [domain] ")[0]
            item['possible_values'] = slot_possible
            item['groundtruth'] = output1
            item['model_output'] = ''
            item['idx'] = dic_key_name
            item['domain_slot_name'] = domain_slot_name
  
            dataset_data[d_name].append(item)
            
        for k,v in dataset_data.items():
            with open(f"../../raw_data/multiwoz/data/MultiWOZ_2.1_preprocess/{data_type}_{k}_LLM_zero-shot.json", 'w') as f:
                json.dump(v, f, indent=4)

    print("done.")
 