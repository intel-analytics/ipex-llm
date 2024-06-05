# BlueLM
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate BlueLM models. For illustration purposes, we utilize the [vivo-ai/BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat) as a reference BlueLM model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a BlueLM model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
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

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the BlueLM model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'vivo-ai/BlueLM-7B-Chat'`.
- `--prompt`: str, argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.4 Sample Output
#### [vivo-ai/BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat)
```log
Inference time: xxxx s
-------------------- Output --------------------
AI是什么？ AI是人工智能（Artificial Intelligence）的缩写，是一种模拟人类智能思维过程的技术。它可以让计算机系统通过学习和适应，自主地完成各种任务，
```

```log
Inference time: xxxx s
-------------------- Output --------------------
What is AI? AI is an AI, or artificial intelligence, that can be defined as the simulation of human intelligence processes by machines, especially computer systems.

AI is not
```
