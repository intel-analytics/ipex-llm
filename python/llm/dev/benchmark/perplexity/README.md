# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation is adapted from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## Environment Preparation
```bash
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install datasets
```
This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

Please set IPEX_LLM_LAST_LM_HEAD=0 to disable the last_lm_head optimization.
```bash
export IPEX_LLM_LAST_LM_HEAD=0
```

## PPL Evaluation
### 1. Run on Wikitext
An example to run perplexity on [wikitext](https://paperswithcode.com/dataset/wikitext-2):
```bash
python run_wikitext.py --model_path meta-llama/Meta-Llama-3-8B --dataset path=wikitext,name=wikitext-2-raw-v1 --precision sym_int4 --device xpu --stride 512 --max_length 4096
```
###  2. Run on [THUDM/LongBench](https://github.com/THUDM/LongBench) dataset

An example to run perplexity on chatglm3-6b using the default Chinese datasets("multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh")
```bash
python run_longbench.py --model_path THUDM/chatglm3-6b --precisions float16 sym_int4 --device xpu --language zh
```


Notes:
- If you want to test model perplexity on a few selected datasets from the `LongBench` dataset, please use the format below.
  ```bash
  --datasets narrativeqa qasper ...
  ```
- The `language` argument will only take effect if `datasets` is `None`. The choices for this argument are `en, zh, all`, which stands for all the English datasets, all the Chinese datasets and all the datasets respectively during testing.
- If you want to test perplexity on pre-downloaded datasets, please specify the `<path/to/dataset>` in the `dataset_path` argument in your command.
- You can run `python make_table.py <input_dir>` to summarize the results.
