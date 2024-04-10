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
pip3 install --pre --upgrade ipex-llm[all]
pip3 install psutil
pip3 install sentencepiece  # Required for LLaMA tokenizer.
pip3 install fastapi
pip3 install "uvicorn[standard]"
pip3 install "pydantic<2"  # Required for OpenAI server.
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
numactl -C 48-95 -m 1 python -m ipex_llm.vllm.entrypoints.openai.api_server \
        --model /MODEL_PATH/Llama-2-7b-chat-hf-ipex/ --port 8000  \
        --load-format 'auto' --device cpu --dtype bfloat16 \
        --load-in-low-bit sym_int4 \
        --max-num-batched-tokens 4096
```

Then you can access the api server as follows:

```bash

 curl http://localhost:8000/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
                 "model": "/MODEL_PATH/Llama-2-7b-chat-hf-ipex/",
                 "prompt": "San Francisco is a",
                 "max_tokens": 128,
                 "temperature": 0
 }' &
```

### 4. (Optional) Add a new model

Currently we have only supported LLaMA family model (including `llama`, `vicuna`, `llama-2`, etc.). To use aother model, you may need add some adaptions.

#### 4.1 Add model code

Create or clone the Pytorch model code to `IPEX/python/llm/src/ipex/llm/vllm/model_executor/models`.

#### 4.2 Rewrite the forward methods

Refering to `IPEX/python/llm/src/ipex/llm/vllm/model_executor/models/ipex_llama.py`, it's necessary to maintain a `kv_cache`, which is a nested list of dictionary that maps `req_id` to a three-dimensional tensor **(the structure may vary from models)**. Before the model's actual `forward` method, you could prepare a `past_key_values` according to current `req_id`, and after that you need to update the `kv_cache` with `output.past_key_values`. The clearence will be executed when the request is finished.

#### 4.3 Register new model

Finally, register your `*ForCausalLM` class to the _MODEL_REGISTRY in `IPEX/python/llm/src/ipex/llm/vllm/model_executor/model_loader.py`.
