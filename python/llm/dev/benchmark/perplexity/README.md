# Perplexity
Perplexity (PPL) is one of the most common metrics for evaluating language models. This benchmark implementation is adapted from [transformers/perplexity](https://huggingface.co/docs/transformers/perplexity#perplexity-of-fixed-length-models) and [benchmark_patch_llm.py](https://github.com/insuhan/hyper-attn/blob/main/benchmark_patch_llm.py) 

## Requirements
To run perplexity test  with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

### 1. Install IPEX
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```


### 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
export BIGDL_LLM_XMX_DISABLED=1
```

</details>

### 4. installing dependency
Install the dataset dependency to download and load dataset for the test.
```bash
pip install datasets
```
## Running the test
### 1.Run on Wikitext
An example to run perplexity on wikitext:
```bash
python run_wikitext.py --model_path meta-llama/Meta-Llama-3-8B --dataset path=wikitext,name=wikitext-2-raw-v1 --precision sym_int4 --device xpu --stride 512 --max_length 4096
```
###  2.Run on [THUDM/LongBench](https://github.com/THUDM/LongBench) dataset

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
