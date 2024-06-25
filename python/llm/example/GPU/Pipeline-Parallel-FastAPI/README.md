# Serve IPEX-LLM on Multiple Intel GPUs in Multi-Stage Pipeline Parallel Fashion

This example demonstrates how to run IPEX-LLM serving on multiple [Intel GPUs](../README.md) with Pipeline Parallel.

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
pip install mpi4py fastapi uvicorn openai
pip install gradio # for gradio web UI
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc

pip install transformers==4.37.0
```

### 2. Run pipeline parallel serving on multiple GPUs

```bash
# Need to set MODEL_PATH in run.sh first
bash run.sh
```


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