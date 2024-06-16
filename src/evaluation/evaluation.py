# 计算JGA指标

import os
import sys
import json
from glob import glob
import argparse
import pandas as pd
import collections
import fire
from config import *
from fuzzywuzzy import fuzz


def fuzzy_string_match(str_ref, str_hyp):
    """Returns fuzzy string similarity score in range [0.0, 1.0]."""

    # The higher the score, the higher the similarity between the two strings.
    return fuzz.token_sort_ratio(str_ref, str_hyp) / 100.0

def noncat_slot_value_match(str_ref_list, str_hyp, use_fuzzy_match):
    """Calculate non-categorical slots correctness.
    Args:
        str_ref_list: a list of reference strings.
        str_hyp: the hypothesis string.
        use_fuzzy_match: whether to use fuzzy string matching.
    Returns:
        score: The highest fuzzy string match score of the references and hypotheis.
    """
    score = 0.0
    for str_ref in str_ref_list:
        if not use_fuzzy_match:
            match_score = float(str_ref == str_hyp)
        else:
            match_score = fuzzy_string_match(str_ref, str_hyp)
        score = max(score, match_score)
    return score


def evaluation(inference_domain,  config_type = 'evaluation'):

    args = generate_config(config_type,inference_domain)
    print(args)

    # load data
    try:
        result_data = json.load(open(args.input_data_dir))
    except:
        result_data = []
        for i in open(args.input_data_dir).readlines():
            try:
                result_data.append(json.loads(i))
            except:
                pass
    
    base_dial_jga_dic = {} #{'idx':True} 只要有一个slot填错，即置false
    optimized_dial_jga_dic = {}

    # base模型的结果
    base_jga_res = {'jga_total':0, 'jga_acc':0, 'total':0, 'acc':0}
    # optmized模型的结果
    optimized_jga_res = {'jga_total':0, 'jga_acc':0, 'total':0, 'acc':0}
    jga_total = 0
    jga_acc = 0
    total = 0
    acc = 0
    domain = ''


    domain = data['domain_slot_name'].split('_')[0]
    base_JGA = float(base_jga_res['jga_acc'])/float(base_jga_res['jga_total'])
    base_ACC = float(base_jga_res['acc'])/float(base_jga_res['total'])
    optimized_JGA = float(optimized_jga_res['jga_acc'])/float(optimized_jga_res['jga_total'])
    optimized_ACC = float(optimized_jga_res['acc'])/float(optimized_jga_res['total'])
    print(f'evalutation base {domain}: JGA: {base_JGA}, ACC: {base_ACC}',{base_jga_res['jga_acc']} ,{base_jga_res['jga_total']},{base_jga_res['total']})
    print(f'evalutation optimized  {domain}: JGA: {optimized_JGA}, ACC: {optimized_ACC}', {optimized_jga_res['jga_acc']} ,{optimized_jga_res['jga_total']},{optimized_jga_res['total']})




if __name__=='__main__':
    fire.Fire(evaluation)
