# ChatGLM
In this directory, you will find examples on how you could apply BigDL-LLM general INT4 optimizations on ChatGLM models. For illustration purposes, we utilize the [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) as a reference ChatGLM model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a ChatGLM model to predict the next N tokens using `generate()` API, with BigDL-LLM general INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latest bigdl-llm nightly build with 'all' option
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the ChatGLM model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/chatglm-6b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-Nano env variables
source bigdl-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
问：AI是什么？
答：
-------------------- Output --------------------
问:AI是什么?
答: AI是人工智能(Artificial Intelligence)的缩写,指的是一种能够模拟人类智能的技术或系统。AI包括机器学习、深度学习、自然语言处理、计算机视觉
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
问：What is AI?
答：
-------------------- Output --------------------
问:What is AI?
答: AI stands for Artificial Intelligence. It refers to the development of computer systems that can perform tasks that typically require human intelligence, such as recognizing speech, understanding natural language
```
