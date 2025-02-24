# vLLM continuous batching on Intel GPUs

This example demonstrates how to serve a LLaMA2-7B model using vLLM continuous batching on Intel GPU (with IPEX-LLM low-bits optimizations).

The code shown in the following example is ported from [vLLM](https://github.com/vllm-project/vllm/tree/v0.6.6).

Currently, we support the following models for vLLM engine:

- Qwen series models
- Llama series models
- ChatGLM series models
- Baichuan series models
- Deepseek series models
- Multimodal models

## Example: Serving LLaMA2-7B using Intel GPU

In this example, we will run Llama2-7b model using Arc A770 and provide `OpenAI-compatible` interface for users.

### 0. Environment

To use Intel GPUs for deep-learning tasks, you should install the XPU driver and the oneAPI Base Toolkit 2025.0.1. Please check the requirements at [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/overview.html).

Besides, you may also want to install the latest compute runtime at [here](https://github.com/intel/compute-runtime/releases)

After install the toolkit, run the following commands in your environment before starting vLLM GPU:
```bash
source /opt/intel/oneapi/setvars.sh
# sycl-ls will list all the compatible Intel GPUs in your environment
sycl-ls

# Example output with one Arc A770:
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.6.32224.500000]
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) w5-3435X OpenCL 3.0 (Build 0) [2024.18.12.0.05_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.52.32224.5]
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
pip install --pre --upgrade "ipex-llm[xpu_2.6]" --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install setuptools-scm
pip install --upgrade cmake
# cd to your workdir
git clone -b 0.6.6 https://github.com/analytics-zoo/vllm.git
cd vllm
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -v /llm/vllm
# For Qwen model support
pip install transformers_stream_generator einops tiktoken
pip install ray
```

### 2. Configure recommended environment variables

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export SYCL_CACHE_PERSISTENT=1
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
export VLLM_RPC_TIMEOUT=100000

 # You may need to adjust the value of
 # --max-model-len, --max-num-batched-tokens, --max-num-seqs
 # to acquire the best performance

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
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
  --tensor-parallel-size 1 \
  --disable-async-output-proc
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

##### Image input

image input only supports [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)now.
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM-V-2_6",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图片里有什么?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 128
  }'
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

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
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
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --disable-async-output-proc
```

### 4. Load low bit models with vLLM

To load low-bit model directly with vLLM, we can use the following option `--low-bit-model-path` when starting service or `low_bit_model_path` when using `vllm_offline_inference.py`.

The low bit model needs to be saved using the `--low-bit-save-path` or `low_bit_save_path` option.

For instance, to save a FP8 low-bit `DeepSeek-R1-Distill-Qwen-7B` model on disk, we can execute the following python script.

```python
from vllm import SamplingParams
from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM

# Create an LLM.
llm = LLM(model="DeepSeek-R1-Distill-Qwen-7B", # Unquantized model path on disk
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="sym_int4",  # The low-bit you may want to quantized to
          tensor_parallel_size=1,      # The tp-size you choose needs to be same when you later uses the low-bit model
          disable_async_output_proc=True,
          distributed_executor_backend="ray",
          max_model_len=500,
          trust_remote_code=True,
          block_size=8,
          max_num_batched_tokens=500,
          low_bit_save_path="/llm/fp8-model-path")  # saved path
```

When finish executing, the low-bit model has been saved at `/llm/fp8-model-path`.

Later we can use the option `--low-bit-model-path /llm/fp8-model-path` to use the low-bit model.


### 5. Known issues

#### Runtime memory

If runtime memory is a concern, you can set --swap-space 0.5 to reduce memory consumption during execution. The default value for --swap-space is 4, which means that by default, the system reserves 4GB of memory for use when GPU memory is insufficient.
