# Serve IPEX-LLM on Multiple Intel GPUs in multi-stage pipeline parallel fashion

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
pip install mpi4py fastapi uvicorn
conda install -c conda-forge -y gperftools=2.10 # to enable tcmalloc

pip install transformers==4.31.0 # for llama2 models
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
