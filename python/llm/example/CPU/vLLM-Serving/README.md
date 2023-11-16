# Serving LLAMA models using vLLM on Intel platforms (experimental support)

This example demonstrates how to serving a llama2-7b model using BigDL-LLM 4 bits optimizations with xeon CPUs with adapted vLLM.

The code shown in the following example is ported from [vLLM](https://github.com/vllm-project/vllm).


## Example: Serving llama2-7b using Xeon CPU

In this example, we will run Llama2-7b model using 48 cores in one socket and provide `OpenAI-compatible` interface for users.

### 1. Install

The original [vLLM](https://github.com/vllm-project/vllm) is designed to run with `CUDA` environment. To adapt vLLM into `Intel` platforms, install the dependencies like this:

```bash
# First create an conda environment
conda create -n bigdl-vllm python==3.9
conda activate bigdl-vllm
# Install dependencies
pip install --pre --upgrade bigdl-llm[all]
pip3 install psutil
pip3 install sentencepiece  # Required for LLaMA tokenizer.
pip3 install numpy
pip3 install "torch==2.0.1"
pip3 install "transformers>=4.33.1"  # Required for Code Llama.
pip3 install "xformers == 0.0.22"
pip3 install fastapi
pip3 install "uvicorn[standard]"
pip3 install "pydantic<2"  # Required for OpenAI server.
```

### 2. Configures Recommending environment variables

```bash
source bigdl-llm-init
```

### 3. Offline inference/Service

#### Offline inference

To run offline inference using vLLM for a quick impression, use the following example:

```bash
#!/bin/bash

# Please first modify the MODEL_PATH in offline_inference.py

numactl -C 48-95 -m 1 python offline_inference.py

```

#### Service

To fully utilize the dynamic batching feature of the `vLLM`, you can send requests to the service using curl or other similar methods.  The requests sent to the engine will be batched at token level. Queries will be executed in the same `forward` step of the LLM and be removed when they are finished instead of waiting all sequences are finished.

```bash
#!/bin/bash
numactl -C 48-95 -m 1 python -m bigdl.llm.vllm.examples.api_server \
        --model /MODEL_PATH/Llama-2-7b-chat-hf-bigdl/ --port 8000  \
        --load-format 'auto' --device cpu --dtype bfloat16
```

Then you can access the api server using the following way:

```bash

 curl http://localhost:8000/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
                 "model": "/MODEL_PATH/Llama-2-7b-chat-hf-bigdl/",
                 "prompt": "San Francisco is a",
                 "max_tokens": 128,
                 "temperature": 0
 }' &
```
### 4. (Optional) Add a new model

Currently we have only supported llama-structure model (including `llama`, `vicuna`, `llama-2`, etc.). To use other model, you may need some adaption to the code.

#### 4.1 Add model code

Create or clone the Pytorch model code to `./models`.

#### 4.2 Rewrite the forward methods

Refering to `./models/bigdl_llama.py`, it's necessary to maintain a `kv_cache`, which is a nested list of dictionary that maps `req_id` to a three-dimensional tensor **(the structure may vary from models)**. Before the model's actual `forward` method, you could prepare a `past_key_values` according to current `req_id`, and after you need to update the `kv_cache` with `output.past_key_values`. The clearence will be executed when the request is finished.

#### 4.3 Register new model

Finally, register your `*ForCausalLM` class to the _MODEL_REGISTRY in `./models/model_loader.py`.