# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation was from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## HOW TO RUN
```python
python run.py --model_path <path/to/model> --precisions sym_int4 fp4 mixed_fp4 sym_int8 fp8_e5m2 fp8_e4m3 mixed_fp8 --device xpu --dataset <path/to/dataset>
```
A more specific example to run perplexity on Llama2-7B and wikitext:
```python
python run.py --model_path meta-llama/Llama-2-7b-chat-hf --precisions float16 sym_int4 --device xpu --dataset /mnt/disk1/datasets
```