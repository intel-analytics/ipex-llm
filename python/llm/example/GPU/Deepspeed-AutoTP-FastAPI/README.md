# Run IPEX-LLM serving on Multiple Intel GPUs using DeepSpeed AutoTP and FastApi

This example demonstrates how to run IPEX-LLM serving on multiple [Intel GPUs](../README.md) by leveraging DeepSpeed AutoTP.

## Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

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
pip install git+https://github.com/microsoft/DeepSpeed.git@ed8aed5
pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@0eb734b
pip install mpi4py fastapi uvicorn
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc
```

> **Important**: IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. Please make sure you have installed the correct version.

### 2. Run tensor parallel inference on multiple GPUs

When we run the model in a distributed manner across two GPUs, the memory consumption of each GPU is only half of what it was originally, and the GPUs can work simultaneously during inference computation.

We provide example usage for `Llama-2-7b-chat-hf` model running on Arc A770

Run Llama-2-7b-chat-hf on two Intel Arc A770:

```bash

# Before run this script, you should adjust the YOUR_REPO_ID_OR_MODEL_PATH in last line
# If you want to change server port, you can set port parameter in last line

# To avoid GPU OOM, you could adjust --max-num-seqs and --max-num-batched-tokens parameters in below script
bash run_llama2_7b_chat_hf_arc_2_card.sh
```

If you successfully run the serving, you can get output like this:

```bash
[0] INFO:     Started server process [120071]
[0] INFO:     Waiting for application startup.
[0] INFO:     Application startup complete.
[0] INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> **Note**: You could change `NUM_GPUS` to the number of GPUs you have on your machine. And you could also specify other low bit optimizations through `--low-bit`.

### 3. Sample Input and Output

We can use `curl` to test serving api

#### generate()

```bash
# Set http_proxy and https_proxy to null to ensure that requests are not forwarded by a proxy.
export http_proxy=
export https_proxy=

curl -X 'POST' \
  'http://127.0.0.1:8000/generate/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "What is AI?",
  "n_predict": 32
}'
```

And you should get output like this:

```json
{
  "index": 0,
  "message": {
    "role": "assistant",
    "content": "\n\nArtificial intelligence (AI) is a branch of computer science that deals with the creation of intelligent machines that can perform tasks that typically "
  },
  "finish_reason": "stop"
}

```
#### generate_stream()
```bash
# Set http_proxy and https_proxy to null to ensure that requests are not forwarded by a proxy.
export http_proxy=
export https_proxy=

curl -X 'POST' \
  'http://127.0.0.1:8000/generate_stream/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "What is AI?",
  "n_predict": 32
}'
```

And you should get output like this:
```json
{"index": 0, "message": {"role": "assistant", "content": "\n"}, "finish_reason": null}
{"index": 1, "message": {"role": "assistant", "content": "\n"}, "finish_reason": null}
{"index": 2, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 3, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 4, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 5, "message": {"role": "assistant", "content": "Artificial "}, "finish_reason": null}
{"index": 6, "message": {"role": "assistant", "content": "intelligence "}, "finish_reason": null}
{"index": 7, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 8, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 9, "message": {"role": "assistant", "content": "(AI) "}, "finish_reason": null}
{"index": 10, "message": {"role": "assistant", "content": "is "}, "finish_reason": null}
{"index": 11, "message": {"role": "assistant", "content": "a "}, "finish_reason": null}
{"index": 12, "message": {"role": "assistant", "content": "branch "}, "finish_reason": null}
{"index": 13, "message": {"role": "assistant", "content": "of "}, "finish_reason": null}
{"index": 14, "message": {"role": "assistant", "content": "computer "}, "finish_reason": null}
{"index": 15, "message": {"role": "assistant", "content": "science "}, "finish_reason": null}
{"index": 16, "message": {"role": "assistant", "content": "that "}, "finish_reason": null}
{"index": 17, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 18, "message": {"role": "assistant", "content": "deals "}, "finish_reason": null}
{"index": 19, "message": {"role": "assistant", "content": "with "}, "finish_reason": null}
{"index": 20, "message": {"role": "assistant", "content": "the "}, "finish_reason": null}
{"index": 21, "message": {"role": "assistant", "content": "creation "}, "finish_reason": null}
{"index": 22, "message": {"role": "assistant", "content": "of "}, "finish_reason": null}
{"index": 23, "message": {"role": "assistant", "content": ""}, "finish_reason": null}
{"index": 24, "message": {"role": "assistant", "content": "intelligent "}, "finish_reason": null}
{"index": 25, "message": {"role": "assistant", "content": "machines "}, "finish_reason": null}
{"index": 26, "message": {"role": "assistant", "content": "that "}, "finish_reason": null}
{"index": 27, "message": {"role": "assistant", "content": "can "}, "finish_reason": null}
{"index": 28, "message": {"role": "assistant", "content": "perform "}, "finish_reason": null}
{"index": 29, "message": {"role": "assistant", "content": "tasks "}, "finish_reason": null}
{"index": 30, "message": {"role": "assistant", "content": "that "}, "finish_reason": null}
{"index": 31, "message": {"role": "assistant", "content": "typically "}, "finish_reason": null}
{"index": 32, "message": {"role": "assistant", "content": null}, "finish_reason": "length"}


```

**Important**: The first token latency is much larger than rest token latency, you could use [our benchmark tool](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/dev/benchmark/README.md) to obtain more details about first and rest token latency.

### 4. Benchmark with wrk

We use wrk for testing end-to-end throughput, check [here](https://github.com/wg/wrk).

You can install by:
```bash
sudo apt install wrk
```

Please change the test url accordingly.

```bash
# set t/c to the number of concurrencies to test full throughput.
wrk -t1 -c1 -d5m -s ./wrk_script_1024.lua http://127.0.0.1:8000/generate/ --timeout 1m
```

## Using the `benchmark.py` Script

The `benchmark.py` script is designed to evaluate the performance of a streaming service by measuring response times and other relevant metrics. Below are the details on how to use the script effectively:

### Command Line Arguments
### Command Line Arguments
- `--prompt_length`: Specifies the length of the prompt used in the test. Acceptable values are `32`, `1024`, and `2048`.
- `--max_concurrent_requests`: Defines the levels of concurrency for the requests. You can specify multiple values to test different levels of concurrency in one run.
- `--max_new_tokens`: Sets the maximum number of new tokens that the model will generate per request. Default is `128`.

### Usage Example
You can run the script with specific settings for prompt length, concurrent requests, and max new tokens by using the following command:

```bash
python benchmark.py --prompt_length 1024 --max_concurrent_requests 1 2 3 --max_new_tokens 128
```

This command sets the prompt length to 1024, tests concurrency levels of 1, 2, and 3, and configures the model to generate up to 128 new tokens per request. The results are saved in log files named according to the concurrency level (1.log, 2.log, 3.log).
