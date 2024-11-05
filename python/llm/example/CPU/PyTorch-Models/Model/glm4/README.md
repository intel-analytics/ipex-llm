# GLM-4
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate GLM-4 models. For illustration purposes, we utilize the [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) as a reference GLM-4 model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a GLM-4 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

# install packages required for GLM-4
pip install "tiktoken>=0.7.0" transformers==4.42.4 "trl<0.12.0"
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

pip install "tiktoken>=0.7.0" transformers==4.42.4 "trl<0.12.0"
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py --prompt 'AI是什么？'
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
numactl -C 0-47 -m 0 python ./generate.py --prompt 'AI是什么？'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the GLM-4 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/glm-4-9b-chat'`.
- `--prompt`: str, argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.4 Sample Output
##### [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)
```log
Inference time: xxxx s
-------------------- Output --------------------

AI是什么？

AI，即人工智能（Artificial Intelligence），是指由人创造出来的，能够模拟、延伸和扩展人的智能的计算机系统或机器。人工智能技术
```

```
Inference time: xxxx s
-------------------- Output --------------------

What is AI?

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term "art
```
