# RWKV

In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate RWKV models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [RWKV/rwkv-4-world-7b](https://huggingface.co/RWKV/rwkv-4-world-7b)  as a reference RWKV model.

## Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a RWKV model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

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

```
python ./generate.py --prompt "你叫什么名字？"
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the RWKV model (e.g. `RWKV/rwkv-4-world-7b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'RWKV/rwkv-4-world-7b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `"你叫什么名字？"`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `40`.

#### Sample Output
#### [RWKV/rwkv-4-world-7b](https://huggingface.co/RWKV/rwkv-4-world-7b)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
Question: 你叫什么名字？

Answer:
-------------------- Output --------------------
Question: 你叫什么名字？

Answer: 我是一个大型语言模型，没有具体的姓名。我是由OpenAI团队创建的，目的是为了提供自然


```
