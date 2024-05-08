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
#!/bin/bash
model="YOUR_MODEL_PATH"
served_model_name="YOUR_MODEL_NAME"

 # You may need to adjust the value of
 # --max-model-len, --max-num-batched-tokens, --max-num-seqs
 # to acquire the best performance

python -m ipex_llm.vllm.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 1
```

You can tune the service using these four arguments:
1. --gpu-memory-utilization: The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9.
2. --max-model-len: Model context length. If unspecified, will be automatically derived from the model config.
3. --max-num-batched-token: Maximum number of batched tokens per iteration.
4. --max-num-seq: Maximum number of sequences per iteration. Default: 256



After the service has been booted successfully, you can send a test request using curl. Here, the `YOUR_MODEL` should be set equal to `$served_model_name` in your booting script.

```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "YOUR_MODEL_NAME",
        "prompt": "San Francisco is a",
        "max_tokens": 128,
        "temperature": 0
 }' &
```

#### Tensor parallel

> Note: We recommend to use docker for tensor parallel deployment.

We have also supported tensor parallel by using multiple XPU cards. To enable tensor parallel, you will need to install `libfabric-dev` in your environment.  In ubuntu, you can install it by:

```bash
sudo apt-get install libfabric-dev
```

To deploy your model across multiple cards, simplely change the value of `--tensor-parallel-size` to the desired value.

For instance, if you have two Arc A770 cards in your environment, then you can set this value to 2. Some OneCCL environment variable settings are also needed, try check the following example:


```bash
#!/bin/bash
model="YOUR_MODEL_PATH"
served_model_name="YOUR_MODEL_NAME"

# CCL needed environment variables
export CCL_WORKER_COUNT=2
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
 # You may need to adjust the value of
 # --max-model-len, --max-num-batched-tokens, --max-num-seqs
 # to acquire the best performance

python -m ipex_llm.vllm.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 2
```
