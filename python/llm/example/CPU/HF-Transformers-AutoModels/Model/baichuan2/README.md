# Baichuan2
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Baichuan2 models. For illustration purposes, we utilize the [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) as a reference Baichuan model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Baichuan model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

pip install transformers_stream_generator  # additional package required for Baichuan-13B-Chat to conduct generation
```

On Windows:
```cmd
onda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install transformers_stream_generator
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Baichuan2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'baichuan-inc/Baichuan2-13B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Baichuan model based on the capabilities of your machine.

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
#### [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<reserved_106> AI是什么？ <reserved_107> 
-------------------- Output --------------------
<reserved_106> AI是什么？ <reserved_107> 人工智能（AI）是指由计算机系统执行的任务，这些任务通常需要人类智能才能完成。AI的目标是使计算机能够模拟人类的思维过程，从而
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<reserved_106> 解释一下“温故而知新” <reserved_107> 
-------------------- Output --------------------
<reserved_106> 解释一下“温故而知新” <reserved_107> 温故而知新是一个成语，出自《论语·为政》篇。这个成语的意思是：通过回顾和了解过去的事情，可以更好地理解新的知识和
```
