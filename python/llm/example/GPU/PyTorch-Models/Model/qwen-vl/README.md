# Qwen-VL
In this directory, you will find examples on how you could use BigDL-LLM `optimize_model` API to accelerate Qwen-VL models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) as a reference Qwen-VL model.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Multimodal chat using `chat()` API
In the example [chat.py](./chat.py), we show a basic use case for a Qwen-VL model to start a multimodal chat using `chat()` API, with BigDL-LLM 'optimize_model' API on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install accelerate tiktoken einops transformers_stream_generator==0.0.4 scipy torchvision pillow tensorboard matplotlib # additional package required for Qwen-VL-Chat to conduct generation
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
```
python ./chat.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Qwen-VL model (e.g `Qwen/Qwen-VL-Chat`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'Qwen/Qwen-VL-Chat'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.
  
In every session, image and text can be entered into cmd (user can skip the input by type **'Enter'**) ; please type **'exit'** anytime you want to quit the dialouge.

Every image output will be named as the round of session and placed under the current directory.

#### Sample Chat
#### [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)

```log
-------------------- Session 1 --------------------
 Please input a picture: http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg
 Please enter the text: 这是什么？
---------- Response ----------
这是一张图片，展现了一个穿着粉色条纹连衣裙的小女孩，她手持一只穿粉色裙子的小熊。这个场景发生在一个户外环境，有砖块背景墙和花朵。

-------------------- Session 2 --------------------
 Please input a picture:
 Please enter the text: 这个小女孩多大了？
---------- Response ----------
根据图片中的描述，这个小女孩应该是年龄较小的孩子，但具体年龄难以确定。从她的外表来看，可能是在5岁左右。。 

-------------------- Session 3 --------------------
 Please input a picture: 
 Please enter the text: 在图中检测框出玩具熊
---------- Response ----------
<ref>玩具熊</ref><box>(330,267),(603,869)</box>

-------------------- Session 4 --------------------
 Please input a picture: exit
```
The sample input image in Session 1 is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>

The sample output image in Session 3 is:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/qwen-vl-example-output-gpu.png"><img width=400px src="https://llm-assets.readthedocs.io/en/latest/_images/qwen-vl-example-output-gpu.png" ></a>
