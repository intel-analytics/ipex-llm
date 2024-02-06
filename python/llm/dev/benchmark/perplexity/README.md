# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation was from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## HOW TO RUN
```python
python run.py --model_path <path/to/model> --precisions sym_int4 fp4 mixed_fp4 sym_int8 fp8_e5m2 fp8_e4m3 mixed_fp8 --device xpu --datasets dataset_name --dataset_path <path/to/dataset> --language en
```
A more specific example to run perplexity on Llama2-7B using the default English datasets:
```python
python run.py --model_path meta-llama/Llama-2-7b-chat-hf --precisions float16 sym_int4 --device xpu 
```

> Note: If you want to test perplexity on your own dataset, please include both the `datasets` and `dataset_path` arguments in your command. We recommend using the `THUDM/LongBench` dataset for testing. If you want to test perplexity on pre-downloaded `THUDM/LongBench` dataset, please specify the `<path/to/LongBench>` in the `dataset_path` argument. The default testing datasets are English datasets, if you want to use the Chinese datasets, please specify in the `language` argument.