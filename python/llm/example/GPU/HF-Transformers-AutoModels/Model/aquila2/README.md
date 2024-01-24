# Aquila2

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Aquila2 models. For illustration purposes, we utilize the [BAAI/AquilaChat2-7B](https://huggingface.co/BAAI/AquilaChat2-7B) as a reference Aquila2 model.

> **Note**: If you want to download the Hugging Face *Transformers* model, please refer to [here](https://huggingface.co/docs/hub/models-downloading#using-git).
>
> BigDL-LLM optimizes the *Transformers* model in INT4 precision at runtime, and thus no explicit conversion is needed.

## Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Aquila2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.

### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
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

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the Aquila2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'BAAI/AquilaChat2-7B'`.
- `--prompt`: str, argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [BAAI/AquilaChat2-7B](https://huggingface.co/BAAI/AquilaChat2-7B)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|startofpiece|>AI是什么？<|endofpiece|>
-------------------- Output --------------------
<|startofpiece|>AI是什么？<|endofpiece|>人工智能（Artificial Intelligence，简称AI）是计算机科学中一个极为重要的研究领域，旨在让计算机模仿人类的智能，包括学习、推理、识别物体
```
