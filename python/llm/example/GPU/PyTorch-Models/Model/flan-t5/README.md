# Flan-t5
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API on Flan-t5 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) as a reference Flan-t5 model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Flan-t5 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
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
python ./generate.py --prompt 'Translate to German: My name is Arthur'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Flan-t5 model (e.g. `google/flan-t5-xxl` to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'google/flan-t5-xxl'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'Translate to German: My name is Arthur'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|User|>:Translate to German: My name is Arthur
-------------------- Output --------------------
Ich bin Arthur.
```
