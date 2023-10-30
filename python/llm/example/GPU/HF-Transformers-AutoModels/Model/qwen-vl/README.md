# Qwen-VL
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Qwen-VL models. For illustration purposes, we utilize the [Qwen/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) as a reference Qwen-VL model.

## Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Multimodal chat using `chat()` API
In the example [chat.py](./chat.py), we show a basic use case for a Qwen-VL model to start a multimodal chat using `chat()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
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
 Please input a picture: https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&q=60&w=500&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8Y2F0fGVufDB8fDB8fHwy
 Please enter the text: 这是什么
---------- Response ----------
图中是一只戴着墨镜的酷炫猫咪，正坐在窗边，看着窗外。 

-------------------- Session 2 --------------------
 Please input a picture: 
 Please enter the text: 这只猫猫多大了？
---------- Response ----------
由于只猫猫戴着太阳镜，无法判断年龄，但可以猜测它应该是一只成年猫猫，已经成年。 

-------------------- Session 3 --------------------
 Please input a picture: 
 Please enter the text: 在图中检测框出猫猫的墨镜
---------- Response ----------
<ref>猫猫的墨镜</ref><box>(398,313),(994,506)</box> 

-------------------- Session 4 --------------------
 Please input a picture: exit
```
The sample input image in Session 1 is (which is fetched from [here](https://images.unsplash.com/photo-1533738363-b7f9aef128ce?auto=format&fit=crop&q=60&w=500&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8Y2F0fGVufDB8fDB8fHwy)):

<a href="https://llm-assets.readthedocs.io/en/latest/_images/qwen-vl-example-input.jpg"><img width=250px src="https://llm-assets.readthedocs.io/en/latest/_images/qwen-vl-example-input.jpg" ></a>

The sample output image in Session 3 is:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/qwen-vl-example-output.png"><img width=250px src="https://llm-assets.readthedocs.io/en/latest/_images/qwen-vl-example-output.png" ></a>
