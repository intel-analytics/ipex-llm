# Serving using IPEX-LLM and vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving. You can find the detailed information at their [homepage](https://github.com/vllm-project/vllm).

IPEX-LLM can be integrated into vLLM so that user can use `IPEX-LLM` to boost the performance of vLLM engine on Intel **GPUs** *(e.g., local PC with descrete GPU such as Arc, Flex and Max)*.


## Quick Start

This quickstart guide walks you through installing and running `vLLM` with `ipex-llm`.

### 1. Install IPEX-LLM for vLLM

IPEX-LLM's support for `vLLM` now is available for only Linux system.

Visit [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html) and follow the instructions in section [Install Prerequisites](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-prerequisites) to isntall prerequisites that are needed for running code on Intel GPUs.

Then,follow instructions in section [Install ipex-llm](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-ipex-llm) to install `ipex-llm[xpu]` and setup the recommended runtime configurations.

**After the installation, you should have created a conda environment, named `ipex-vllm` for instance, for running `vLLM` commands with IPEX-LLM.**

### 2. Install vLLM

Currently, we maintain a specific branch of vLLM, which only works on Intel GPUs. 

Activate the `ipex-vllm` conda environment and install vLLM by execcuting the commands below.

```bash
conda activate ipex-vllm
source /opt/intel/oneapi/setvars.sh
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

**Now you are all set to use vLLM with IPEX-LLM**

## 3. Offline inference/Service

### Offline inference

To run offline inference using vLLM for a quick impression, use the following example.

```eval_rst
.. note::

  Please modify the MODEL_PATH in offline_inference.py to use your chosen model. 
  You can try modify load_in_low_bit to different values in **[sym_int4, fp8, fp16]** to use different quantization dtype.
```

```bash
#!/bin/bash
wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/python/llm/example/GPU/vLLM-Serving/offline_inference.py
python offline_inference.py
```

For instructions on how to change the `load_in_low_bit` value in `offline_inference.py`, check the following example:

```bash
llm = LLM(model="YOUR_MODEL",
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          # Simply change here for the desired load_in_low_bit value
          load_in_low_bit="sym_int4",
          tensor_parallel_size=1,
          trust_remote_code=True)
```

The result of executing `Baichuan2-7B-Chat` model with `sym_int4` low-bit format is shown as follows:

```
Prompt: 'Hello, my name is', Generated text: ' [Your Name] and I am a [Your Job Title] at [Your'
Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government in the United States. The president leads'
Prompt: 'The capital of France is', Generated text: ' Paris.\nThe capital of France is Paris.'
Prompt: 'The future of AI is', Generated text: " bright, but it's not without challenges. As AI continues to evolve,"
```

### Service
To fully utilize the continuous batching feature of the `vLLM`, you can send requests to the service using `curl` or other similar methods. The requests sent to the engine will be batched at token level. Queries will be executed in the same `forward` step of the LLM and be removed when they are finished instead of waiting for all sequences to be finished.


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

1. `--gpu-memory-utilization`: The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9.
2. `--max-model-len`: Model context length. If unspecified, will be automatically derived from the model config.
3. `--max-num-batched-token`: Maximum number of batched tokens per iteration.
4. `--max-num-seq`: Maximum number of sequences per iteration. Default: 256

If the service have been booted successfully, the console will display messages similar to the following:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" width=100%; />
</a>


After the service has been booted successfully, you can send a test request using `curl`. Here, `YOUR_MODEL` should be set equal to `$served_model_name` in your booting script, e.g. `Qwen1.5`.


```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "YOUR_MODEL",
  "prompt": "San Francisco is a",
  "max_tokens": 128,
  "temperature": 0
}' | jq '.choices[0].text'
```

Below shows an example output using `Qwen1.5-7B-Chat` with low-bit format `sym_int4`:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/vllm-curl-result.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/vllm-curl-result.png" width=100%; />
</a>

```eval_rst
.. tip::

  If your local LLM is running on Intel Arc™ A-Series Graphics with Linux OS (Kernel 6.2), it is recommended to additionaly set the following environment variable for optimal performance before starting the service:

  .. code-block:: bash

      export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

```

## 4. About Tensor parallel

> Note: We recommend to use docker for tensor parallel deployment. Check our serving docker image `intelanalytics/ipex-llm-serving-xpu`.

We have also supported tensor parallel by using multiple Intel GPU cards. To enable tensor parallel, you will need to install `libfabric-dev` in your environment. In ubuntu, you can install it by:

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

If the service have booted successfully, you should see the output similar to the following figure:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" width=100%; />
</a>

## 5.Performing benchmark

To perform benchmark, you can use the **benchmark_throughput** script that is originally provided by vLLM repo.

```bash
conda activate ipex-vllm

source /opt/intel/oneapi/setvars.sh

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

wget https://raw.githubusercontent.com/intel-analytics/ipex-llm/main/docker/llm/serving/xpu/docker/benchmark_vllm_throughput.py -O benchmark_throughput.py

export MODEL="YOUR_MODEL"

# You can change load-in-low-bit from values in [sym_int4, fp8, fp16]

python3 ./benchmark_throughput.py \
    --backend vllm \
    --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $MODEL \
    --num-prompts 1000 \
    --seed 42 \
    --trust-remote-code \
    --enforce-eager \
    --dtype float16 \
    --device xpu \
    --load-in-low-bit sym_int4 \
    --gpu-memory-utilization 0.85
```

The following figure shows the result of benchmarking `Llama-2-7b-chat-hf` using 50 prompts:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/vllm-benchmark-result.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/vllm-benchmark-result.png" width=100%; />
</a>


```eval_rst
.. tip::

  To find the best config that fits your workload, you may need to start the service and use tools like `wrk` or `jmeter` to perform a stress tests.
```