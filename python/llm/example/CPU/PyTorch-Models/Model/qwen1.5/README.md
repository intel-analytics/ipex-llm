# Qwen1.5
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate Qwen1.5 models. For illustration purposes, we utilize the [Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) as reference Qwen1.5 model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Qwen1.5 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.37.0 # install transformers which supports Qwen2

# only for Qwen1.5-MoE-A2.7B
pip install transformers==4.40.0
pip install trl==0.8.1
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install transformers==4.37.0

REM for Qwen1.5-MoE-A2.7B
pip install transformers==4.40.0
pip install trl==0.8.1
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Qwen1.5 to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Qwen/Qwen1.5-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output
#### [Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
AI是什么？<|im_end|>
<|im_start|>assistant
-------------------- Output --------------------
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
AI是什么？<|im_end|>
<|im_start|>assistant
AI（Artificial Intelligence）是指由计算机程序实现的智能，它使机器能够模拟人类的思考、学习和决策过程，从而解决各种复杂
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
-------------------- Output --------------------
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
AI, or artificial intelligence, refers to the simulation of human intelligence in machines that are programmed to think, learn, and make decisions like humans. It involves the
```
