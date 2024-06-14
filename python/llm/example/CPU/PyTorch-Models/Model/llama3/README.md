# Llama3
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate Llama3 models. For illustration purposes, we utilize the [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) as a reference Llama3 model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Llama3 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

# transformers>=4.33.0 is required for Llama3 with IPEX-LLM optimizations
pip install transformers==4.37.0
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

pip install transformers==4.37.0
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py --prompt 'What is AI?'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py --prompt 'What is AI?'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama3 model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Meta-Llama-3-8B-Instruct'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.4 Sample Output
#### [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is AI?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


-------------------- Output (skip_special_tokens=False) --------------------
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is AI?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that would typically require human intelligence, such as:

1. Learning: AI
```