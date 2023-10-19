# Qwen-VL-Chat
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate Qwen-VL-Chat models. For illustration purposes, we utilize the [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) as a reference Qwen-VL-Chat model.

## Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Multimodal chat using `chat()` API
In the example [generate.py](./generate.py), we show a basic use case for a Qwen-VL-Chat model to start a multimodal chat using `chat()` API, with BigDL-LLM 'optimize_model' API.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latest bigdl-llm nightly build with 'all' option
pip install accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed torchvision matplotlib
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
python ./chat.py
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-Nano env variables
source bigdl-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./chat.py
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the Qwen-VL-Chat model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Qwen/Qwen-VL-Chat'`.
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `32`.
  
In every session, image and text can be entered into cmd (user can skip the input by type **'Enter'**) ; please type **'exit'** anytime you want to quit the dialouge.

Every image output will be named as the round of session and placed under the current directory.

#### 2.4 Sample Chat
#### [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
```log
-------------------- Session 1 --------------------
 请输入你想询问的图片: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
 请输入你想询问的文字信息: 这是什么
---------- Response ----------
图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它们处于沙滩上。

-------------------- Session 2 --------------------
 请输入你想询问的图片:
 请输入你想询问的文字信息: 输出击掌的检测框
---------- Response ----------
<ref>击掌</ref><box>(538,513),(588,601)</box>

-------------------- Session 3 --------------------
 请输入你想询问的图片: exit
```

[![Demo.gif](https://i.postimg.cc/0N1QshjQ/Demo.gif)](https://postimg.cc/yDnBhQF4)