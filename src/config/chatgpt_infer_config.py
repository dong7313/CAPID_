import os


def gen_chatgpt_infer_config(inference_domain, *args, **kwargs):
    from config import global_args, DotDict, ensure_directory

   
    API_KEY = ''
    chatgpt_input_file = ensure_directory(f'../../raw_data/multiwoz/data/MultiWOZ_2.1_preprocess/test_{inference_domain}_LLM_zero-shot.json')

    chatgpt_output_file = ensure_directory(f'../../raw_data/multiwoz/data/MultiWOZ_2.1_chatgpt/base/test_{inference_domain}_chat_LLM_zero-shot.json')


    ctx_prompt = """instruction: "{}"

    #     dialogue: "{}"

    # groundtruth:
    # "{}"

    # model output:
    # "{}"

    # Compare the groundtruth and model output from these aspects: correctness (if the response follows the instruction correctly and give an accurate response, high priority), helpfulness(like depth, creativity, coherence). Then be an expert prompt engineer and improve my instruction by changing "So the value of slot <##> is" into a question from the above aspects to get better responses like "groundtruth" rather than "model output".

    # Pay attention to:
    # 1.Change the sentence in the instruction:"So the value of slot <##> is" into a question, make sure your new question ask for the same thing with the slot <##>. make sure it is smooth and reasonable, Replace the slot "<##>" with some information in the dialogue.
    # 2.Firstly, determine whether the slot <##> is mentioned in the dialogue, if not, output "Whether the slot is mentioned in the dialogue:" with No.
    # 3.If the slot <##> is mentioned in the dialogue, cross over the question with the dialogue information, make sure the question contains adjectival phrases from the dialogue for the groundtruth noun, you need to make sure the adjectival phrases are really contained in the question
    # 4.If the slot <##> is not mentioned in the dialogue, try to give some hints in the question that may guide the question to tell the the slot does not exist.
    # 5.If the slot <##> is mentioned in the dialogue, you might change the slot name it into some other description using the dialogue information , here is an example:"So the value of slot <attraction-area> is", is changed into " Based on the conversation in [DIALOGUE_CONTEXT], can you determine the specific area of Cambridge the user is interested in visiting?"
    # 5.Replace the whole dialuge with a token:"[DIALOGE_CONTEXT]".
    # 6.You should never generate a response to the original instruction!
    # 7.Whether the slot <##> is mentioned in the dialogue, you should keep Sentence pattern consistent, like"can you determine the specific area of Cambridge the user is interested in visiting".


    # Output with the following format:
    # Whether the slot is mentioned in the dialogue: Yes/No [END]
    # Optimized Instruction: xxx [END]"""


     # for other backbone model
#     ctx_prompt = """Instruction: 
   
#     Slot: "{slot}"
#     Dialogue: "{dialgoue}"
#     Groundtruth:"{groundtruth}"

# Be an expert prompt engineer and enhance the original slot {slot} by transforming it into a question that has the same meaning with {slot} effectively solicits the ground truth, utilizing the context information related to the {groundtruth}.
    
# Pay attention to:
#     1.Replace the slot {slot} with synonyms in the dialogue, if any  
#     2.If the slot {slot} is mentioned in the dialogue, cross over the question with the dialogue context information. Ensure that the question includes adjectival phrases from the dialogue that describe the ``groundtruth''. It is crucial that these adjectival phrases are accurately incorporated into the question to maintain context relevance and accuracy. If not, do nothing.
#     3.Replace the whole dialuge with a token:``[DIALOGE\_CONTEXT]''
#     4.You should never generate a response to the original instruction!
    
# Output with the following format:
#     Valuable context information from the dialogue: xxx [END]
#     Optimized Instruction: xxx [END]

 
#     """
    chatgpt_info_args  = DotDict({
        'API_KEY' : API_KEY,
        'HEADERS' : {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"},
        'API_URL': "",
        'chatgpt_output_file':chatgpt_output_file,
        'chatgpt_input_file':chatgpt_input_file,
        'ctx_prompt':ctx_prompt
    })
    return chatgpt_info_args