# vLLM continuous batching on Intel CPUs (experimental support)

This example demonstrates how to serve a LLaMA2-7B model using vLLM continuous batching on Intel CPU (with IPEX-LLM 4 bits optimizations).

The code shown in the following example is ported from [vLLM](https://github.com/vllm-project/vllm/tree/v0.2.1.post1).

## Example: Serving LLaMA2-7B using Xeon CPU

In this example, we will run Llama2-7b model using 48 cores in one socket and provide `OpenAI-compatible` interface for users.

### 1. Install

To run vLLM continuous batching on Intel CPUs, install the dependencies as follows:

```bash
# First create an conda environment
conda create -n ipex-vllm python=3.11
conda activate ipex-vllm
# Install dependencies
pip3 install numpy
pip3 install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install psutil
pip3 install sentencepiece  # Required for LLaMA tokenizer.
pip3 install fastapi
pip3 install "uvicorn[standard]"
pip3 install "pydantic<2"  # Required for OpenAI server.

# Install vllm
git clone https://github.com/vllm-project/vllm.git && \
cd ./vllm && \
git checkout v0.4.2 && \
pip install wheel packaging ninja setuptools>=49.4.0 numpy && \
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
VLLM_TARGET_DEVICE=cpu python3 setup.py install
```

### 2. Configure recommended environment variables

```bash
source ipex-llm-init -t
```

### 3. Offline inference/Service

#### Offline inference

To run offline inference using vLLM for a quick impression, use the following example:

```bash
#!/bin/bash

# Please first modify the MODEL_PATH in offline_inference.py
# Modify load_in_low_bit to use different quantization dtype

numactl -C 48-95 -m 1 python offline_inference.py

```

#### Service

To fully utilize the continuous batching feature of the `vLLM`, you can send requests to the service using curl or other similar methods.  The requests sent to the engine will be batched at token level. Queries will be executed in the same `forward` step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.

```bash
#!/bin/bash
# You may also want to adjust the `--max-num-batched-tokens` argument, it indicates the hard limit
# of batched prompt length the server will accept
numactl -C 48-95 -m 1 python -m ipex_llm.vllm.cpu.entrypoints.openai.api_server \
        --model /MODEL_PATH/Llama-2-7b-chat-hf/ --port 8000  \
        --load-format 'auto' --device cpu --dtype bfloat16 \
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
