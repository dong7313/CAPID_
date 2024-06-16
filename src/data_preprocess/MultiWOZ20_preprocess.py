
import os
import sys
import json
from glob import glob
from random import sample

domain_desc_flag = True # To append domain descriptions or not 
slot_desc_flag = True  # To append slot descriptions or not 
PVs_flag = True # for categorical slots, append possible values as suffix

def preprocess(dial_json, schema, out, idx_out, domain_list, frame_idxs, test_json_list, val_json_list, split, multi_type):
  dial_json_n = dial_json.split("/")[-1]
  dial_json = open(dial_json)
  dial_json = json.load(dial_json)

  count = 0
  for dial_idx in dial_json.keys():

    if split == "train":
        if dial_idx in test_json_list or dial_idx in val_json_list:
            continue
    elif split == "dev":
        if dial_idx not in val_json_list :
            continue
    elif split == "test":
        if dial_idx not in test_json_list :
            continue
    else:
        print("error")
        sys.exit(1)

    dial = dial_json[dial_idx]['log']

    cur_dial = ""

    for turn_id in range(len(dial)):
      content = dial[turn_id]
      if len(content['metadata']) == 0:
        speaker = " [USER] " 
    
      else:
        speaker = " [SYSTEM] "   
      uttr = content['text']
      cur_dial += speaker
      cur_dial += uttr  

      if len(content['metadata']) > 0:
        active_slot_values = {}
        turn_slot_dic = content['metadata']
        frame_key_flag = set()
        for frame_key in turn_slot_dic.keys():
          if frame_key not in domain_list:
              continue
    
          for slot_act in turn_slot_dic[frame_key]['semi'].keys():
              if turn_slot_dic[frame_key]['semi'][slot_act] != "":
                active_slot_values[frame_key.lower()+"-"+slot_act.lower()] = turn_slot_dic[frame_key]['semi'][slot_act] # 这也不一样
                frame_key_flag.add(frame_key.lower())
      
          for slot_act in turn_slot_dic[frame_key]['book'].keys():
              if slot_act != "booked" and turn_slot_dic[frame_key]['book'][slot_act] != "":
                active_slot_values[frame_key.lower()+'-'+slot_act.lower()] = turn_slot_dic[frame_key]['book'][slot_act] # 这也不一样
                frame_key_flag.add(frame_key.lower())


        for domain in schema.keys():
   
          if multi_type == '2.0':

            slots = [domain.split("-")[1]]
            if slots.startswith('book'):
               slots =slots.replace("book", "")
            d_name = domain.split("-")[0]
          elif multi_type == '2.1':
            slots = [domain.split("-")[2]]
            d_name = domain.split("-")[0]
          for slot in slots:
            s_name = slot
            # generate schema prompt w/ or w/o natural langauge descriptions
            schema_prompt = ""
            schema_prompt += " [domain] " + d_name + ","
            schema_prompt += " [slot] " + s_name + "."
            if PVs_flag:
              # only append possible values if the slot is categorical
              if len(schema[domain]) > 0:
                PVs = ", ".join(schema[domain])
                #print(PVs)
                #sys.exit(1)
                schema_prompt += " [Possible Values] " + PVs
            #print("schema prompt:", schema_prompt)
            #sys.exit(1)
            domain_slot = d_name.lower() + "-" + s_name.lower()
            s_name2 = s_name
            while " " in s_name2:
                s_name2 = s_name2.replace(" ", "")
            domain_slot2 = d_name.lower() + "-" + s_name2.lower()
            if domain_slot in active_slot_values.keys():
              target_value = active_slot_values[domain_slot]
              # if target_value == 'not mentioned':
              #    print(active_slot_values)
              ##print(target_value)
              #sys.exit()
            elif domain_slot2 in active_slot_values.keys():
              target_value = active_slot_values[domain_slot2]

            else:
              target_value = "not mentioned"
   
            if target_value == 'not mentioned':
              target_value = 'NONE'
            line = { "dialogue": cur_dial + schema_prompt, "state":  target_value }
            
            out.write(json.dumps(line))
            out.write("\n")
            idx_list = [ dial_json_n, str(dial_idx), str(turn_id), str(frame_idxs[d_name]), d_name, s_name ]
            idx_out.write("|||".join(idx_list))
            idx_out.write("\n")
      count +=1


  return


def Generate_data(data_path, domain_list, multi_type):



    parent_dir = os.path.dirname(os.path.dirname(data_path))
    multiwoz_dir = os.path.basename(os.path.dirname(data_path))
    data_path_out = os.path.join(parent_dir, multiwoz_dir+"_preprocess")
    if not os.path.exists(data_path_out):
      os.makedirs(data_path_out)
    print(data_path_out)
    #sys.exit(1)

    frame_idxs = {}
    for service_idx in range(len(domain_list)):
        service = domain_list[service_idx]
        frame_idxs[service] = service_idx

    
    if multi_type == '2.0':
      test_list_file = open(os.path.join(data_path, "{}.json".format("testListFile")))
      val_list_file = open(os.path.join(data_path, "{}.json".format("valListFile")))
    elif multi_type == '2.1':
      test_list_file = open(os.path.join(data_path, "{}.txt".format("testListFile")))
      val_list_file = open(os.path.join(data_path, "{}.txt".format("valListFile")))
    test_json_list = []
    for lin in test_list_file:
        test_json_list.append(lin.strip())
    
    
    val_json_list = []
    for lin in val_list_file:
        val_json_list.append(lin.strip())

    
    for split in ["train","dev", "test"]:
    #for split in [ "test"]:
        print("--------Preprocessing {} set---------".format(split))
        out = open(os.path.join(data_path_out, "{}.json".format(split)), "w") # W 模式 会覆盖之前的内容
        idx_out = open(os.path.join(data_path_out, "{}.idx".format(split)), "w")
        dial_jsons = glob(os.path.join(data_path, "*json"))
        
        schema_path = data_path + "ontology.json"
        schema = json.load(open(schema_path))

        for dial_json in dial_jsons:
            if "data.json" in dial_json:
                preprocess(dial_json, schema, out, idx_out, domain_list, frame_idxs, test_json_list, val_json_list, split, multi_type)
        idx_out.close()
        out.close()
    print("--------Finish Preprocessing---------")


# 先统计一下训练集和测试集中的service有哪些区别
def Analysis(data_path):


    schema_path = data_path + "ontology.json"
    schema = json.load(open(schema_path))


    domain_list = []
    slot_list = []

    for domain in schema.keys():
        #print(domain['service_name'])
        d = domain.split("-")[0]
        slot_list.append(domain)
        if d not in domain_list:
            domain_list.append(d)
   
    print(domain_list)
       
    print(slot_list)

    print(len(domain_list))



    return domain_list

def main():

    data_path = "../../raw_data/multiwoz/data/MultiWOZ_2.1/"
    domain_list = Analysis(data_path)
    print(domain_list)
    multi_type = '2.1'
    Generate_data(data_path, domain_list, multi_type)

if __name__=='__main__':
    main()