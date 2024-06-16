import os

def gen_evaluation(inference_domain):
    from config import global_args,DotDict,domain_split
    dataset_type = '../../raw_data/multiwoz/data/MultiWOZ_2.1'
    dataset_version = 'MultiWOZ_2.1'
    base_model_type = 't5'
    batch_size = 16
    micro_batch_size = 4
    num_epochs = 5
    input_data_dir = os.path.join(f'{dataset_type}_inference/result/', base_model_type, '_'.join(map(str, ['test', inference_domain, batch_size, micro_batch_size, num_epochs])))
    evaluation_output_dir = os.path.join('../../evaluation', '_'.join([inference_domain, base_model_type]))
    evaluation_info_args = DotDict({
        'input_data_dir':input_data_dir, 
        'evaluation_output_dir':evaluation_output_dir,
    })
    return evaluation_info_args
