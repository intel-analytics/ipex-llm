# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation was from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## HOW TO RUN
```bash
python run.py --model_path <path/to/model> --precisions sym_int4 fp4 mixed_fp4 sym_int8 fp8_e5m2 fp8_e4m3 mixed_fp8 --device xpu --datasets dataset_names --dataset_path <path/to/dataset> --language en
```
A more specific example to run perplexity on Llama2-7B using the default English datasets:
```bash
python run.py --model_path meta-llama/Llama-2-7b-chat-hf --precisions float16 sym_int4 --device xpu --language en
```

> Note: We currently only support the `THUDM/LongBench` [dataset](https://github.com/THUDM/LongBench)

- If you want to test model perplexity on a few selected datasets from the `LongBench` dataset, please use the format below.
  ```bash
  --datasets narrativeqa qasper ...
  ```
- The `language` argument will only take effect if `datasets` is `None`. The choices for this argument are `en, zh, all`, which stands for all the English datasets, all the Chinese datasets and all the datasets respectively during testing.
- If you want to test perplexity on pre-downloaded datasets, please specify the `<path/to/dataset>` in the `dataset_path` argument in your command.

## Summarize the results
```python
python make_table.py <input_dir>
```
