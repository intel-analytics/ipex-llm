# Mistral
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Mistral models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as reference Mistral models.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

**Important: According to [Mistral Troubleshooting](https://huggingface.co/mistralai/Mistral-7B-v0.1#troubleshooting), please make sure you have installed `transformers==4.34.0` to run the example.**

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Mistral model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

# Refer to https://huggingface.co/mistralai/Mistral-7B-v0.1#troubleshooting, please make sure you are using a stable version of Transformers, 4.34.0 or newer.
pip install transformers==4.34.0
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Mistral model (e.g. `mistralai/Mistral-7B-Instruct-v0.1` and `mistralai/Mistral-7B-v0.1`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'mistralai/Mistral-7B-Instruct-v0.1'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output
#### [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
```log
Inference time: xxxx s
-------------------- Output --------------------
[INST] What is AI? [/INST] AI stands for Artificial Intelligence. It is a branch of computer science that focuses on the development of intelligent machines that work, react, and even think like humans
```

#### [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
```log
Inference time: xxxx s
-------------------- Output --------------------
[INST] What is AI? [/INST]

[INST] Artificial Intelligence (AI) is a branch of computer science that deals with the simulation of intelligent behavior in computers. It is a broad
```
