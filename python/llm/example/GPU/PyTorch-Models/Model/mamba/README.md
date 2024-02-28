# Mamba
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate Mamba models. For illustration purposes, we utilize the [state-spaces/mamba-1.4b](https://huggingface.co/state-spaces/mamba-1.4b) and [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) as reference Mamba models.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Mamba model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install einops # package required by Mamba
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
python ./generate.py
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Mamba model (e.g `state-spaces/mamba-1.4b` and `state-spaces/mamba-2.8b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `state-spaces/mamba-1.4b`.
- `--tokenizer-repo-id-or-path`: argument defining the huggingface repo id for the tokenizer of Mamba model to be downloaded, or the path to the huggingface checkpoint folder.  It is default to be `EleutherAI/gpt-neox-20b`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output
#### [state-spaces/mamba-1.4b](https://huggingface.co/state-spaces/mamba-1.4b)
```log
Inference time: xxxx s
-------------------- Output --------------------
What is AI?

Artificial Intelligence (AI) is a broad term that describes the use of artificial intelligence (AI) to create artificial intelligence (AI). AI is a
```

#### [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b)
```log
Inference time: xxxx s
-------------------- Output --------------------
What is AI?

Artificial Intelligence is a field of study that focuses on creating machines that can perform intelligent functions. These functions are performed by machines that are smarter than humans
```
