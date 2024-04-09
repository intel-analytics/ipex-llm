# InternLM2
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate InternLM2 models. For illustration purposes, we utilize the [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) as reference InternLM2 model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a InternLM2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for IPEX-LLM:
```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] # install the latest ipex-llm nightly build with 'all' option
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the InternLM2 to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'internlm/internlm2-chat-7b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output
#### [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|User|>:AI是什么？
<|Bot|>:
-------------------- Output --------------------
<|User|>:AI是什么？
<|Bot|>:AI是人工智能的缩写，是计算机科学的一个分支，旨在使计算机能够像人类一样思考、学习和执行任务。AI技术包括机器学习、自然
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|User|>:What is AI?
<|Bot|>:
-------------------- Output --------------------
<|User|>:What is AI?
<|Bot|>:AI is the ability of machines to perform tasks that would normally require human intelligence, such as perception, reasoning, learning, and decision-making. AI is made possible
```
