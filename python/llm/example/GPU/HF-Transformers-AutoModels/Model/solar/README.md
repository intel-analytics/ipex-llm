# SOLAR-10.7B
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on SOLAR-10.7B models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) as a reference SOLAR-10.7B model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a SOLAR-10.7B model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers==4.35.2 # required by SOLAR-10.7B
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
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the SOLAR-10.7B model (e.g `upstage/SOLAR-10.7B-Instruct-v1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'upstage/SOLAR-10.7B-Instruct-v1.0'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [upstage/SOLAR-10.7B-Instruct-v1.0t](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)

```log
Inference time: XXXX s
-------------------- Prompt --------------------
<s>### User:
What is AI?
### Assistant:

-------------------- Output --------------------
### User:
What is AI?
### Assistant:
 AI, or Artificial Intelligence, refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It involves the development of
```

```log
Inference time: XXXX s
-------------------- Prompt --------------------
<s>### User:
AI是什么？
### Assistant:

-------------------- Output --------------------
### User:
AI是什么？
### Assistant:
AI, 全称为人工智能(Artificial Intelligence)，是计算机科学、心理学、语言学、逻
```
