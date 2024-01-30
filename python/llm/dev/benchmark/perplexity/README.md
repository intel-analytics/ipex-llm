# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation was from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [llm_perplexity.py](https://github.com/luo-cheng2021/ov.cpu.llm.experimental/blob/main/llm_perplexity.py) 

## HOW TO RUN
```python
python run.py --model_path <path/to/model> --precisions sym_int4 fp4 mixed_fp4 sym_int8 fp8_e5m2 fp8_e4m3 mixed_fp8 --device xpu --dataset path=<dataset_path>,name=<dataset_name>
```
A more specific example to run perplexity on Llama2-7B and wikitext:
```python
python run.py --model_path meta-llama/Llama-2-7b-chat-hf --precisions float16 sym_int4 --device xpu --dataset path=wikitext,name=wikitext-2-raw-v1
```