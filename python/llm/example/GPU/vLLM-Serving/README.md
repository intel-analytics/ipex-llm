# vLLM continuous batching on Intel GPUs (experimental support)

This example demonstrates how to serve a LLaMA2-7B model using vLLM continuous batching on Intel GPU (with IPEX-LLM low-bits optimizations).

The code shown in the following example is ported from [vLLM](https://github.com/vllm-project/vllm/tree/v0.3.3).

Currently, we support the following models for vLLM engine:

- Qwen series models
- Llama series models
- ChatGLM series models
- Baichuan series models

## Example: Serving LLaMA2-7B using Intel GPU

In this example, we will run Llama2-7b model using Arc A770 and provide `OpenAI-compatible` interface for users.

### 0. Environment

To use Intel GPUs for deep-learning tasks, you should install the XPU driver and the oneAPI Base Toolkit 2024.0. Please check the requirements at [here](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU#requirements).

After install the toolkit, run the following commands in your environment before starting vLLM GPU:
```bash
source /opt/intel/oneapi/setvars.sh
# sycl-ls will list all the compatible Intel GPUs in your environment
sycl-ls

# Example output with one Arc A770:
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```

### 1. Install

Install the dependencies for vLLM as follows:

```bash
# This directory may change depends on where you install oneAPI-basekit
source /opt/intel/oneapi/setvars.sh
# First create an conda environment
conda create -n ipex-vllm python=3.11
conda activate ipex-vllm
# Install dependencies
pip install --pre --upgrade "ipex-llm[xpu]" --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# cd to your workdir
git clone -b sycl_xpu https://github.com/analytics-zoo/vllm.git
cd vllm
pip install -r requirements-xpu.txt
pip install --no-deps xformers
VLLM_BUILD_XPU_OPS=1 pip install --no-build-isolation -v -e .
pip install outlines==0.0.34 --no-deps
pip install interegular cloudpickle diskcache joblib lark nest-asyncio numba scipy
# For Qwen model support
pip install transformers_stream_generator einops tiktoken
```

### 2. Configure recommended environment variables

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
### 3. Offline inference/Service

#### Offline inference

To run offline inference using vLLM for a quick impression, use the following example:


```bash
#!/bin/bash

# Please first modify the MODEL_PATH in offline_inference.py
# Modify load_in_low_bit to use different quantization dtype
python offline_inference.py
```

#### Service

To fully utilize the continuous batching feature of the `vLLM`, you can send requests to the service using curl or other similar methods.  The requests sent to the engine will be batched at token level. Queries will be executed in the same `forward` step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.

For vLLM, you can start the service using the following command:
```bash
python -m ipex_llm.vllm.entrypoints.openai.api_server \
        --model /MODEL_PATH/Llama-2-7b-chat-hf/ --port 8000  \
        --device xpu --dtype float16 \
        --load-in-low-bit sym_int4 \
        --max-num-batched-tokens 4096
```


Then you can access the api server as follows:

```bash

 curl http://localhost:8000/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
                 "model": "/MODEL_PATH/Llama-2-7b-chat-hf/",
                 "prompt": "San Francisco is a",
                 "max_tokens": 128,
                 "temperature": 0
 }' &
```
