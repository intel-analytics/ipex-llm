# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation is adapted from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## Run on Wikitext

Download the dataset from [here](https://paperswithcode.com/dataset/wikitext-2), unzip it and we will use the test dataset `wiki.test.raw` for evaluation.

```bash
python run_wikitext.py --model_path meta-llama/Meta-Llama-3-8B/ --data_path wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw --precision sym_int4 --use-cache --device xpu

# Run with stride
python run_wikitext.py --model_path meta-llama/Meta-Llama-3-8B/ --data_path wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw --precision fp16 --device xpu --stride 512
```

## Run on [THUDM/LongBench](https://github.com/THUDM/LongBench) dataset

```bash
python run.py --model_path <path/to/model> --precisions sym_int4 fp8 --device xpu --datasets dataset_names --dataset_path <path/to/dataset> --language en
```
A more specific example to run perplexity on Llama2-7B using the default English datasets:
```bash
python run.py --model_path meta-llama/Llama-2-7b-chat-hf --precisions float16 sym_int4 --device xpu --language en
```

Notes:
- If you want to test model perplexity on a few selected datasets from the `LongBench` dataset, please use the format below.
  ```bash
  --datasets narrativeqa qasper ...
  ```
- The `language` argument will only take effect if `datasets` is `None`. The choices for this argument are `en, zh, all`, which stands for all the English datasets, all the Chinese datasets and all the datasets respectively during testing.
- If you want to test perplexity on pre-downloaded datasets, please specify the `<path/to/dataset>` in the `dataset_path` argument in your command.
- You can run `python make_table.py <input_dir>` to summarize the results.
