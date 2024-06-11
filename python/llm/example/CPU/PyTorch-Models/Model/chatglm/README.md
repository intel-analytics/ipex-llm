# ChatGLM
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate ChatGLM models. For illustration purposes, we utilize the [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) as a reference ChatGLM model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a ChatGLM model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:


On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install "transformers<4.34.1"
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install "transformers<4.34.1"
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

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the ChatGLM model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm-6b'`.
- `--prompt`: str, argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.4 Sample Output
#### [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
```log
Inference time: xxxx s
-------------------- Output --------------------
问:AI是什么?
答: AI是人工智能(Artificial Intelligence)的缩写,指的是一种能够模拟人类智能的技术或系统。AI包括机器学习、深度学习、自然语言处理、计算机视觉
```
