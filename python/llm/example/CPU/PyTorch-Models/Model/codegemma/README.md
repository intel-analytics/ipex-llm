# CodeGemma
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate CodeGemma models. For illustration purposes, we utilize the [google/codegemma-7b-it](https://huggingface.co/google/codegemma-7b-it) as reference CodeGemma models.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a CodeGemma model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for IPEX-LLM:
```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all]

# According to CodeGemma's requirement, please make sure you are using a stable version of Transformers, 4.38.1 or newer.
pip install transformers==4.38.1
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
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
numactl -C 0-47 -m 0 python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the CodeGemma model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'google/codegemma-7b-it'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'Write a hello world program'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.4 Sample Output
#### [google/codegemma-7b-it](https://huggingface.co/google/codegemma-7b-it)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<bos><start_of_turn>user
Write a hello world program<end_of_turn>
<start_of_turn>model

-------------------- Output --------------------
<start_of_turn>user
Write a hello world program<end_of_turn>
<start_of_turn>model
```python
print("Hello, world!")
```

This program will print the message "Hello, world!" to the console.
```
