# Dolly v2
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate Dolly v2 models. For illustration purposes, we utilize the [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b) and [databricks/dolly-v2-7b](https://huggingface.co/databricks/dolly-v2-7b) as reference Dolly v2 models.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Dolly v2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Dolly v2 model (e.g. `databricks/dolly-v2-12b` and `databricks/dolly-v2-7b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'databricks/dolly-v2-12b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output

#### [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)
```log
Inference time: xxxx s
-------------------- Output --------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is AI?

### Response:
Artificial Intelligence (AI) is a term generally used to describe computer systems that can perform tasks that typically require human intelligence. AI has a broad range of applications
```

#### [databricks/dolly-v2-7b](https://huggingface.co/databricks/dolly-v2-7b)
```log
Inference time: xxxx s
-------------------- Output --------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is AI?

### Response:
Artificial Intelligence (AI) is a field of computer science, artificial intelligence, and robotics that focuses on understanding and mastering the principles of intelligence and making
```
