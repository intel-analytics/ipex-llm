# Replit
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Replit models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [replit/replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) as a reference Replit model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an Replit model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
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
python ./generate.py --prompt 'def print_hello_world():'
```
More information about arguments can be found in [Arguments Info](#31-arguments-info) section. The expected output can be found in [Sample Output](#32-sample-output) section.

#### 3.1 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Replit model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'replit/replit-code-v1-3b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'def print_hello_world():'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 3.2 Sample Output
#### [replit/replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
def print_hello_world():
-------------------- Output --------------------
def print_hello_world():
    print("Hello")
    print("World")

print_hello_world()


def print_hello_world():
    print
```