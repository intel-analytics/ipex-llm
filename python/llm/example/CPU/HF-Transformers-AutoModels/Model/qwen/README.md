# Qwen

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Qwen models. For illustration purposes, we utilize the [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) as a reference Qwen model.

## 0. Requirements

To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a Qwen model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.

### 1. Install

We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

pip install "transformers<4.37.0"
pip install tiktoken einops transformers_stream_generator  # additional package required for Qwen-7B-Chat to conduct generation
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

pip install "transformers<4.37.0"
pip install tiktoken einops transformers_stream_generator
```

### 2. Run

The minimum Qwen model version currently supported by IPEX-LLM is the version on November 30, 2023.

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Qwen model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Qwen/Qwen-7B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Qwen model based on the capabilities of your machine.

#### 2.1 Client

On client Windows machine, it is recommended to run directly with full utilization of all cores:

```cmd
python ./generate.py 
```

#### 2.2 Server

For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,

```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output

#### [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>AI是什么？ <bot>
-------------------- Output --------------------
<human>AI是什么？ <bot>AI，也称为人工智能，是指计算机科学的一个分支，其目标是创造出能够执行某些任务的智能机器。AI的研究涵盖了机器学习、深度学习
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>What is AI? <bot>
-------------------- Output --------------------
<human>What is AI? <bot>AI stands for Artificial Intelligence. It refers to the ability of a computer program or machine to perform tasks that typically require
```
