# Llama2
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate Llama2 models. For illustration purposes, we utilize the [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) as reference Llama2 models.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example 1 - Basic Version: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```bash
python ./generate.py --prompt 'What is AI?'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI is a field of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as understanding natural language,
```

#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI, or artificial intelligence, refers to the ability of machines to perform tasks that would typically require human intelligence, such as learning, problem-solving,
```

## Example 2 - Low Memory Version: Predict Tokens using `generate()` API

If you're not able to load the full 4-bit model (e.g. [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)) in one GPU as shown in [Example 1](#example-1---basic-version-predict-tokens-using-generate-api), you may try this example instead.

In [low_memory_generate.py](./low_memory_generate.py), we show a way to load very large models with very low GPU memory footprint. However this could be much slower than the standard way. The implementation is adapted from [here](https://www.kaggle.com/code/simjeg/platypus2-70b-without-wikipedia-rag).

### 1. Environment setup
Please refer to [Example 1](#example-1---basic-version-predict-tokens-using-generate-api) for more information.

### 2. Run

```bash
python ./low_memory_generate.py --split-weight --splitted-weights-path ${SPLITTED_WEIGHTS_PATH}
```

In the example, besides arguments in [Example 1](#3-run), several other arguments can be passed to satisfy your requirements:

- `--splitted-weights-path`: argument defining folder saving per-layer weights.
- `--split-weight`: argument defining whether to split weights by layer. If this argument is enabled, per-layer weights will be generated and saved to `--splitted-weights-path`. This argument only needs to be enabled once for the same model.
- `--max-cache-num`: argument defining the maximum number of weights saved in the cache. You can adjust this argument based on your GPU memory. It is default to be 200. For [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf), GPU peak memory is around 3G when it is set to 0 and 15G when it is set to 200.
