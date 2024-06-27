# Run IPEX-LLM serving on Multiple Intel GPUs using DeepSpeed AutoTP and FastApi

This example demonstrates how to run IPEX-LLM serving on multiple [Intel GPUs](../../../python/llm/example/GPU/README.md) by leveraging DeepSpeed AutoTP.

## Table of Contents
- [Requirements](./deepspeed_autotp_fastapi_quickstart.md#requirements)
- [Example](./deepspeed_autotp_fastapi_quickstart.md#example)


## Requirements

To run this example with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../python/llm/example/GPU/README.md#requirements) for more information. For this particular example, you will need at least two GPUs on your machine.

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

> [!IMPORTANT]
> IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. Please make sure you have installed the correct version.

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

> [!NOTE]
> You could change `NUM_GPUS` to the number of GPUs you have on your machine. And you could also specify other low bit optimizations through `--low-bit`.

### 3. Sample Input and Output

We can use `curl` to test serving api

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
  "generated_text": "What is AI? Artificial intelligence (AI) refers to the development of computer systems able to perform tasks that would normally require human intelligence, such as visual perception, speech",
  "generate_time": "0.45149803161621094s"
}

```

> [!IMPORTANT]
> The first token latency is much larger than rest token latency, you could use [our benchmark tool](../../../python/llm/dev/benchmark/README.md) to obtain more details about first and rest token latency.

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