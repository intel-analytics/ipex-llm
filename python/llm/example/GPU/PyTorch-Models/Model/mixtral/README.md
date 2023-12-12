# Mixtral
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate Mixtral models. For illustration purposes, we utilize the DiscoResearch/mixtral-7b-8expert(https://huggingface.co/DiscoResearch/mixtral-7b-8expert) as a reference Mixtral model.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

**Important: Please make sure you have installed `transformers==4.36.0` to run the example.**

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Mixtral model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

# Please make sure you are using a stable version of Transformers, 4.36.0 or newer.
pip install transformers==4.36.0
```

### 2. Download Model and Replace File
To run [DiscoResearch/mixtral-7b-8expert](https://huggingface.co/DiscoResearch/mixtral-7b-8expert) model on Intel GPU, we have provided an updated version [DiscoResearch-mixtral-7b-8expert/modeling_moe_mistral.py](./DiscoResearch-mixtral-7b-8expert/modeling_moe_mistral.py) of `modeling_moe_mistral.py`.


#### 2.1 Download Model
You could use the following code to download [DiscoResearch/mixtral-7b-8expert](https://huggingface.co/DiscoResearch/mixtral-7b-8expert).

```python
from huggingface_hub import snapshot_download

# for DiscoResearch/mixtral-7b-8expert
model_path = snapshot_download(repo_id='DiscoResearch/mixtral-7b-8expert')
print(f'DiscoResearch/mixtral-7b-8expert checkpoint is downloaded to {model_path}')
```

#### 2.2 Replace `modeling_moe_mistral.py`
For `DiscoResearch/mixtral-7b-8expert`, you should replace the `modeling_moe_mistral.py` with [DiscoResearch-mixtral-7b-8expert/modeling_moe_mistral.py](./DiscoResearch-mixtral-7b-8expert/modeling_moe_mistral.py).

### 3. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```bash
python ./generate.py --prompt 'What is AI?'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Mixtral model (e.g. `DiscoResearch/mixtral-7b-8expert`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'DiscoResearch/mixtral-7b-8expert'`. For model `DiscoResearch/mixtral-7b-8expert`, you should input the path to the model folder in which `modeling_moe_mistral.py` has been replaced.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [DiscoResearch/mixtral-7b-8expert](https://huggingface.co/DiscoResearch/mixtral-7b-8expert)
```log
Inference time: xxxx s
-------------------- Output --------------------
[INST] What is AI? [/INST]

[INST] Artificial Intelligence (AI) is the ability of a computer program or a machine to think and learn. It is also a field of
```
