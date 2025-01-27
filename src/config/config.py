import os
import json
import torch
import time
import logging
from transformers import AutoTokenizer
import torch.nn.functional as F

from inference_t5_prompt_config import *
from finetune_t5_prompt_config import *
from chatgpt_infer_config import *
from finetune_t5_result_config import *
from inference_result_t5_config import *
from evaluation_config import *

# Set device globally
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass

# Setting up logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DotDict(dict):
    """Dictionary subclass that allows dot notation access to its items."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

global_args = DotDict({
    'multi20_domain_dict': {'hotel': 0, 'train': 1, 'attraction': 2, 'restaurant': 3, 'taxi': 4}
})

def ensure_directory(path):
    """Ensures the directory exists, creating it if necessary."""
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    logging.info(f"Directory checked/created: {path_dir}")
    return path

def generate_config(config_type, inference_domain, prompt_domain=None, prompt_type='test', idx=1, *args, **kwargs):
    """Generates configurations based on the provided type."""
    config_map = {
        'finetune_t5_prompt': gen_finetune_t5_prompt,
        'inference_t5_prompt': gen_inference_t5_prompt,
        'finetune_t5_result': gen_finetune_t5_result,
        'inference_result_t5': gen_inference_result_t5,
        'evaluation': gen_evaluation,
        'chatgpt_infer': gen_chatgpt_infer_config
    }
    
    if config_type not in config_map:
        raise ValueError(f"Invalid config type: {config_type}")
    
    return config_map[config_type](inference_domain, prompt_domain, prompt_type, idx)

def contrastive_decoding_noncausal(slot_domain, data, model, tokenizer, interpolation=0.9, max_length=50):
    """Performs contrastive decoding for non-causal models."""
    prompt_temp = """
[INST]1.Transform the original slot {slot} into a clearer, user-friendly question that seeks the same information with {slot}
2.Enrich the new question with the relevant context in [Dialogue], like adjectival phrases for the domain or synonyms for the slot [/INST]. \n[Dialogue]:{dialogue}"""
    prompt_temp_cd = """
[INST]1.Craft a question that incorporates both the domain and the slot found within the dialogue.
2.Enrich the new question with the relevant context in [Dialogue], like adjectival phrases for the domain or synonyms for the slot. [/INST] \n[Dialogue]:{dialogue}"""

    input_seq1 = prompt_temp.format(slot=data['domain_slot_name'], dialogue=data['dialogue']) + "\n Optimized Prompt: "
    input_seq2 = prompt_temp_cd.format(dialogue=data['dialogue']) + "\n Optimized Prompt: "

    input_ids1 = tokenizer(input_seq1, return_tensors='pt', max_length=1024).input_ids.to(device)
    input_ids2 = tokenizer(input_seq2, return_tensors='pt', max_length=1024).input_ids.to(device)

    decoder_start_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(device)

    decoded_sequence = ""
    for _ in range(max_length):
        with torch.no_grad():
            outputs1 = model(input_ids=input_ids1, decoder_input_ids=decoder_input_ids)
            outputs2 = model(input_ids=input_ids2, decoder_input_ids=decoder_input_ids)

            next_token_logits1 = outputs1.logits[:, -1, :]
            next_token_logits2 = outputs2.logits[:, -1, :]
            prob1 = F.log_softmax(next_token_logits1, dim=-1)
            prob2 = F.log_softmax(next_token_logits2, dim=-1)

            mask = torch.zeros_like(prob2, dtype=torch.bool)
            mask[0][prob2.argmax(dim=-1).item()] = 0.01
            prob2 *= mask

            debiased_prob = prob1 - interpolation * prob2
            next_token_id = debiased_prob.argmax(dim=-1, keepdim=True)

        decoder_input_ids = torch.cat((decoder_input_ids, next_token_id), dim=1)
        decoded_sequence = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return decoded_sequence

def get_latest_checkpoint(checkpoint_dirs):
    """Retrieves the latest checkpoint from the specified directory."""
    pro_files = os.listdir(checkpoint_dirs)
    latest_steps = 0
    latest_checkpoint = None

    for dir in pro_files:
        try:
            with open(os.path.join(checkpoint_dirs, dir, 'trainer_state.json'), 'r') as f:
                state = json.load(f)
                steps = state.get("global_step", 0)
                if steps > latest_steps:
                    latest_steps = steps
                    latest_checkpoint = dir
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error reading {dir}: {e}")
            continue
    
    if latest_checkpoint is None:
        raise ValueError("No valid checkpoints found.")
    
    return os.path.join(checkpoint_dirs, latest_checkpoint)

logging.info('End of loading config.py')
