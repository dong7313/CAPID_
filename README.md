# CAPID: Zero-shot Cross-domain Dialogue State Tracking via Context-aware Auto-prompting and Instruction-following Contrastive Decoding
## Data preparation
We use the data processing script provided by [DST-as-Prompting](https://github.com/chiahsuan156/DST-as-Prompting) for data pre-processing, post-processing and evaluation.

Benchmark Dataset can be downloaded at:
MultiWOZ: https://github.com/budzianowski/multiwoz

```ruby
conda create -n audst python=3.8
conda activate audst
pip install -r requirements.txt
```
## Dataset Preparation
```ruby
cd raw_data/
git clone https://github.com/budzianowski/multiwoz ./
unzip MultiWOZ_2.1.zip
cd src/data_preprocess/
python MultiWOZ20_preprocess.py
python data_prepare_zeroshot_MultiWOZ20.py
```

## Auto-prompting
```ruby
cd src/autoprompt/
python autoprompt.py
```

## Finetuning For Student Model
We choose T5-base as the backbone model for student model due to inference efficiency. Please specify your model path in src/config/finetune_t5_prompt_config.py

```ruby
cd src/finetune/
CUDA_VISIBLE_DEVICES=0 python finetune_t5_prompt.py --inference_domain hotel

```

## Inference For Student Model
If you want to test on Hotel domain, then perform inference setting "inference_domain == hotel & prompt_domain == hotel & prompt_type test" and " == hotel & prompt_domain == {other domain} & prompt_type train" and 
```ruby
cd src/inference/
CUDA_VISIBLE_DEVICES=0 python inference_t5_prompt.py --inference_domain hotel --prompt_domain hotel --prompt_type test
```


## Finetuning For DST Model
```ruby
cd src/finetune/
CUDA_VISIBLE_DEVICES=0 python fintune_t5_result.py --inference_domain hotel  ## for t5 Model
CUDA_VISIBLE_DEVICES=0 python finetune_llama_result.py --inference_domain hotel  ## for LLaMa Model
```

## Inference For DST Model
```ruby
cd src/inference/
CUDA_VISIBLE_DEVICES=0 python inference_result_t5.py --inference_domain hotel # for t5 Model
CUDA_VISIBLE_DEVICES=0  python inference_llama_results.py --inference_domain hotel # for LLama Model
```


## Evaluation
```ruby
cd src/evaluation/
CUDA_VISIBLE_DEVICES=0 python evaluation.py --inference_domain hotel
```