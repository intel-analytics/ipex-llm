# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation is adapted from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## Run on Wikitext

Setting the --dataset parameter to 'path=wikitext,name=wikitext-2-raw-v1' can fetch the wikitext dataset. The parameter --max_length is used to limit the maximum length of the model to avoid Out-Of-Memory. Here is a specific example to run on wikitext.
```bash

python run_wikitext.py --model_path meta-llama/Meta-Llama-3-8B --dataset path=wikitext,name=wikitext-2-raw-v1 --precision sym_int4 --device xpu --stride 512 --max_length 4096

```

## Run on [THUDM/LongBench](https://github.com/THUDM/LongBench) dataset

```bash
python run_longbench.py --model_path <path/to/model> --precisions sym_int4 fp8 --device xpu --datasets dataset_names --dataset_path <path/to/dataset> --language en
```
A more specific example to run perplexity on Llama2-7B using the default English datasets:
```bash
python run_longbench.py --model_path meta-llama/Llama-2-7b-chat-hf --precisions float16 sym_int4 --device xpu --language en
```

Notes:
- If you want to test model perplexity on a few selected datasets from the `LongBench` dataset, please use the format below.
  ```bash
  --datasets narrativeqa qasper ...
  ```
- The `language` argument will only take effect if `datasets` is `None`. The choices for this argument are `en, zh, all`, which stands for all the English datasets, all the Chinese datasets and all the datasets respectively during testing.
- If you want to test perplexity on pre-downloaded datasets, please specify the `<path/to/dataset>` in the `dataset_path` argument in your command.
- You can run `python make_table.py <input_dir>` to summarize the results.
