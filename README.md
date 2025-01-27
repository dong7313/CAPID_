# CAPID: Zero-shot Cross-domain Dialogue State Tracking via Context-aware Auto-prompting and Instruction-following Contrastive Decoding
Thank you for your interest in our work, and this is the original implementation of "Zero-shot Cross-domain Dialogue State Tracking via Context-aware Auto-prompting and Instruction-following Contrastive Decoding".
## Data preparation
We use the data processing script provided by [DST-as-Prompting](https://github.com/chiahsuan156/DST-as-Prompting) for data pre-processing, post-processing and evaluation.

Benchmark Dataset can be downloaded at:
- **MultiWOZ**: [https://github.com/budzianowski/multiwoz](https://github.com/budzianowski/multiwoz)


```bash
conda create -n audst python=3.8
conda activate audst
pip install -r requirements.txt
```
## Dataset Preparation
```bash
cd raw_data/
git clone https://github.com/budzianowski/multiwoz ./
unzip MultiWOZ_2.1.zip
cd src/data_preprocess/
python MultiWOZ20_preprocess.py
python data_prepare_zeroshot_MultiWOZ20.py
```
> Note: The following instructions are based on using the hotel domain as a zero-shot test domain.
## Auto-prompting
We provide 1000 examples for hotel domain in `raw_data/multiwoz/data/MultiWOZ_2.1_chatgpt/base/test_hotel_chat_LLM_zero-shot.json`. These examples have been generated using auto-prompting.
```bash
cd src/autoprompt/
python autoprompt.py --inference_domain hotel
```

## Finetuning For Student Model
We have chosen T5-base as the backbone model for the student model due to its inference efficiency. Please specify the path to your T5-base model in `src/config/finetune_t5_prompt_config.py`
```bash
cd src/finetune/
CUDA_VISIBLE_DEVICES=0 python finetune_t5_prompt.py --inference_domain hotel

```

## Inference For Student Model
To test on the hotel domain, set the following parameters:
```python
inference_domain == hotel
prompt_domain == hotel
prompt_type == test
```
Additionally, for cross-domain training, set:
```python
inference_domain == hotel
prompt_domain == {other domain}
prompt_type == train
```
```bash
cd src/inference/
CUDA_VISIBLE_DEVICES=0 python inference_t5_prompt.py --inference_domain hotel --prompt_domain hotel --prompt_type test
```


## Finetuning For DST Model
We provide the training parameters for LoRA in the hotel domain at `checkpoints/MultiWOZ_2.1/result_finetune/llama2_exp/hotel_128_4_1/`
```ruby
cd src/finetune/
CUDA_VISIBLE_DEVICES=0 python fintune_t5_result.py --inference_domain hotel  ## for t5 Model
CUDA_VISIBLE_DEVICES=0 python finetune_llama_result.py --inference_domain hotel  ## for LLaMa Model
```

## Inference For DST Model
```bash
cd src/inference/
CUDA_VISIBLE_DEVICES=0 python inference_result_t5.py --inference_domain hotel # for t5 Model
CUDA_VISIBLE_DEVICES=0  python inference_llama_results.py --inference_domain hotel # for LLama Model
```


## Evaluation
```bash
cd src/evaluation/
CUDA_VISIBLE_DEVICES=0 python evaluation.py --inference_domain hotel
```

## Citation
If this work proves beneficial or use our code for your research, citing our paper would be greatly appreciated.
```bash
@inproceedings{dong2024zero,
  title={Zero-shot Cross-domain Dialogue State Tracking via Context-aware Auto-prompting and Instruction-following Contrastive Decoding},
  author={Dong, Xiaoyu and Feng, Yujie and Lu, Zexin and Shi, Guangyuan and Wu, Xiao-Ming},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={8527--8540},
  year={2024}
}
```
