# Serving using IPEX-LLM and vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving. You can find the detailed information at their [homepage](https://github.com/vllm-project/vllm).

IPEX-LLM can be integrated into vLLM so that user can use `IPEX-LLM` to boost the performance of vLLM engine on Intel GPUs.


## Quick Start

This quickstart guide walks you through installing and running `vLLM` with `ipex-llm`.

## 1. Install IPEX-LLM

IPEX-LLM's support for `vLLM` now is available for Linux system with Intel GPUs.

Visit [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html) and follow the instructions to install prerequisites for running code on XPU and `IPEX-LLM[xpu]`

## 2. Install vLLM

Currently, we maintain a specific branch of vLLM, which only works on Intel GPU. You can follow the following instructions to install vLLM:

```bash
# Please activate your conda environment first.
source /opt/intel/oneapi/setvars.sh
# Install dependencies
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

## 3. Offline inference/Service

### Offline inference

To run offline inference using vLLM for a quick impression, use the following example. The `offline_inference.py` script can be acquired at [here](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/offline_inference.py).

```bash
#!/bin/bash

# Please first modify the MODEL_PATH in offline_inference.py
# Modify load_in_low_bit to use different quantization dtype
python offline_inference.py
```

### Service
To fully utilize the continuous batching feature of the `vLLM`, you can send requests to the service using curl or other similar methods. The requests sent to the engine will be batched at token level. Queries will be executed in the same `forward` step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.


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

After the service has been booted successfully, you can send a test request using `curl`. Here, `YOUR_MODEL` should be set equal to `$served_model_name` in your booting script.


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

## 4. About Tensor parallel

> Note: We recommend to use docker for tensor parallel deployment. Check our docker image intelanalytics/ipex-llm-serving-xpu

We have also supported tensor parallel by using multiple XPU cards. To enable tensor parallel, you will need to install `libfabric-dev` in your environment. In ubuntu, you can install it by:

```bash
sudo apt-get install libfabric-dev
```

To deploy your model across multiple cards, simplely change the value of `--tensor-parallel-size` to the desired value.


For instance, if you have two Arc A770 cards in your environment, then you can set this value to 2. Some OneCCL environment variable settings are also needed, check the following example:

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

## 5. Integrate with FastChat Service

vLLM can also be integrated into FastChat as an backend serving engine. In this section, we will show how to use vLLM async engine with FastChat.

### Installation

To use FastChat with vLLM, you will need to install `ipex-llm` before installing vLLM according to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/fastchat_quickstart.html#install-ipex-llm-with-fastchat).

Then you can install `vLLM` normally as introduced in previous sections.

### Using the vLLM worker

You can refer to this [document](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/fastchat_quickstart.html#start-the-service) on how to use the FastChat framework.

To use vLLM as an serving backend for FastChat, you need to start the `controller` normally