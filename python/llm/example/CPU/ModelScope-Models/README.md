# Run ModelScope Model

In this directory, you will find example on how you could apply IPEX-LLM INT4 optimizations on ModelScope models. For illustration purposes, we utilize the [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary) as a reference ModelScope model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a ChatGLM3 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
# Refer to https://github.com/modelscope/modelscope/issues/765, please make sure you are using 1.11.0 version
pip install modelscope==1.11.0
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install modelscope==1.11.0
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the ModelScope repo id for the ModelScope ChatGLM3 model to be downloaded, or the path to the ModelScope checkpoint folder. It is default to be `'ZhipuAI/chatglm3-6b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the ChatGLM3 model based on the capabilities of your machine.

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
#### [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|user|>
AI是什么？
<|assistant|>
-------------------- Output --------------------
[gMASK]sop <|user|>
AI是什么？
<|assistant|> AI是人工智能（Artificial Intelligence）的缩写，指的是通过计算机程序和算法模拟人类智能的技术。AI可以帮助我们解决各种问题，例如语音
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|user|>
What is AI?
<|assistant|>
-------------------- Output --------------------
[gMASK]sop <|user|>
What is AI?
<|assistant|>
AI stands for Artificial Intelligence. It refers to the development of computer systems that can perform tasks that would normally require human intelligence, such as recognizing speech or making
```
