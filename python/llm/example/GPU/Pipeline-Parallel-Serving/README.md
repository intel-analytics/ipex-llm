# Serve IPEX-LLM on Multiple Intel GPUs in Multi-Stage Pipeline Parallel Fashion

This example demonstrates how to run IPEX-LLM serving on multiple [Intel GPUs](../README.md) with Pipeline Parallel.

## Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

## Verified Models
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B-Instruct
- codellama/CodeLlama-13b-hf
- Qwen/Qwen2-7B-Instruct
- Qwen/Qwen1.5-7B-Chat
- Qwen/Qwen1.5-14B-Chat
- Qwen/Qwen1.5-32B-Chat
- Qwen/Qwen1.5-MoE-A2.7B-Chat
- Qwen/CodeQwen1.5-7B-Chat
- THUDM/chatglm3-6b
- THUDM/glm-4-9b-chat
- baichuan-inc/Baichuan2-7B-Chat
- baichuan-inc/Baichuan2-13B-Chat
- microsoft/Phi-3-mini-4k-instruct
- mistralai/Mistral-7B-v0.1
- mistralai/Mixtral-8x7B-Instruct-v0.1
- 01-ai/Yi-6B-Chat

## Example

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh
pip install mpi4py fastapi uvicorn openai
pip install gradio # for gradio web UI
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc

pip install transformers==4.37.0

# only for Qwen1.5-MoE-A2.7B
pip install transformers==4.40.0
pip install trl==0.8.1
```

### 2-1. Run ipex-llm serving on one GPU card 

Refer to [Lightweight-Serving](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Lightweight-Serving), get the [lightweight_serving.py](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Lightweight-Serving/lightweight_serving.py) and then to run.
```bash
# Need to set NUM_GPUS=1 and MODEL_PATH in run.sh first
bash run.sh
```

### 2-2. Run pipeline parallel serving on multiple GPUs

```bash
# Need to set MODEL_PATH in run.sh first
bash run.sh
```

### Command Line Arguments in `run.sh`
> Note: INT4 optimization is applied to the model by default. You could specify other low bit optimizations (such as 'fp8' and 'fp6') through `--low-bit`. Besides, you could change `NUM_GPUS` to the number of GPUs you have on your machine. Other relative settings are listed below:

- `--low-bit`: Sets the low bit optimizations (such as 'sym_int4', 'fp16', 'fp8' and 'fp6') for the model.
- `--max-num-seqs`: Sets the maximum batch size on a single card during pipeline parallel serving.
- `--max-prefilled-seqs`: Sets the maximum batch size for prefilled sequences. Use `0` to disable partial prefetching and process all requests in a single batch.

### 3. Sample Input and Output

We can use `curl` to test serving api

#### generate()

```bash
# Set no_proxy to ensure that requests are not forwarded by a proxy.
export no_proxy=localhost,127.0.0.1

curl -X POST -H "Content-Type: application/json" -d '{
  "inputs": "What is AI?",
  "parameters": {
    "max_new_tokens": 32
  },
  "stream": false
}' http://localhost:8000/generate
```


### 4. Benchmark with wrk

We use wrk for testing end-to-end throughput, check [here](https://github.com/wg/wrk).

You can install by:
```bash
sudo apt install wrk
```

Please change the test url accordingly.

```bash
# set t/c to the number of concurrencies to test full throughput.
wrk -t1 -c1 -d5m -s ./wrk_script_1024.lua http://127.0.0.1:8000/generate --timeout 1m
```

## 5. Using the `benchmark.py` Script

The `benchmark.py` script is designed to evaluate the performance of a streaming service by measuring response times and other relevant metrics. Below are the details on how to use the script effectively:

### Command Line Arguments

- `--prompt_length`: Specifies the length of the prompt used in the test. Acceptable values are `32`, `128`, `1024`, and `2048`.
- `--max_concurrent_requests`: Defines the levels of concurrency for the requests. You can specify multiple values to test different levels of concurrency in one run.
- `--max_new_tokens`: Sets the maximum number of new tokens that the model will generate per request. Default is `128`.

### Usage Example
You can run the script with specific settings for prompt length, concurrent requests, and max new tokens by using the following command:

```bash
python benchmark.py --prompt_length 1024 --max_concurrent_requests 1 2 3 --max_new_tokens 128
```

This command sets the prompt length to 1024, tests concurrency levels of 1, 2, and 3, and configures the model to generate up to 128 new tokens per request. The results are saved in log files named according to the concurrency level (1.log, 2.log, 3.log).

## 6. Gradio Web UI

```bash
python ./gradio_webui.py -m Llama-2-13b-chat-hf 
```