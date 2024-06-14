# ChatGLM2

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on ChatGLM2 models. For illustration purposes, we utilize the [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) as a reference ChatGLM2 model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## 1. Install
We suggest using conda to manage environment:

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

## 2. Run

### Example 1: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a ChatGLM2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the ChatGLM2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm2-6b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the ChatGLM2 model based on the capabilities of your machine.

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
##### [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
问：AI是什么？

答：
-------------------- Output --------------------
问：AI是什么？

答： AI指的是人工智能,是一种能够通过学习和推理来执行任务的计算机程序。它可以模仿人类的思维方式,做出类似人类的决策,并且具有自主学习、自我
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
问：What is AI?

答：
-------------------- Output --------------------
问：What is AI?

答： Artificial Intelligence (AI) refers to the ability of a computer or machine to perform tasks that typically require human-like intelligence, such as understanding language, recognizing patterns
```

### Example 2: Stream Chat using `stream_chat()` API
In the example [streamchat.py](./streamchat.py), we show a basic use case for a ChatGLM2 model to stream chat, with IPEX-LLM INT4 optimizations.

**Stream Chat using `stream_chat()` API**:
```
python ./streamchat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --question QUESTION
```

**Chat using `chat()` API**:
```
python ./streamchat.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --question QUESTION --disable-stream
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the ChatGLM2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm2-6b'`.
- `--question QUESTION`: argument defining the question to ask. It is default to be `"晚上睡不着应该怎么办"`.
- `--disable-stream`: argument defining whether to stream chat. If include `--disable-stream` when running the script, the stream chat is disabled and `chat()` API is used.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the ChatGLM2 model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
$env:PYTHONUNBUFFERED=1  # ensure stdout and stderr streams are sent straight to terminal without being first buffered
python ./streamchat.py
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
export PYTHONUNBUFFERED=1  # ensure stdout and stderr streams are sent straight to terminal without being first buffered
numactl -C 0-47 -m 0 python ./streamchat.py
```
