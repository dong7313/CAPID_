# CAPID: Zero-shot Cross-domain Dialogue State Tracking via Context-aware Auto-prompting and Instruction-following Contrastive Decoding
## Data preparation
We use the data processing script provided by [DST-as-Prompting](https://github.com/chiahsuan156/DST-as-Prompting) for data pre-processing, post-processing and evaluation.
```ruby
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
```ruby
cd src/finetune/
python fintune_t5_prompt.py --inference_domain hotel
```

## Inference For Student Model
```ruby
cd src/inference/
python inference_t5_prompt.py --inference_domain hotel
```


## Finetuning For DST Model
```ruby
cd src/finetune/
python fintune_t5_result.py --inference_domain hotel
```

## Inference For DST Model
```ruby
cd src/inference/
python inference_result_t5.py --inference_domain hotel
```


## Evaluation
```ruby
cd src/evaluation/
python evaluation.py --inference_domain hotel
```