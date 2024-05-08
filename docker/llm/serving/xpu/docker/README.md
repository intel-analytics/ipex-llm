## Build/Use IPEX-LLM-serving xpu image

### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-serving-xpu:2.1.0-SNAPSHOT .
```


### Use the image for doing xpu serving


To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container.

An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:2.1.0-SNAPSHOT

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```


After the container is booted, you could get into the container through `docker exec`.

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```
After the container is booted, you could get into the container through `docker exec`.

Currently, we provide two different serving engines in the image, which are FastChat serving engine and vLLM serving engine.

#### FastChat serving engine

To run model-serving using `IPEX-LLM` as backend using FastChat, you can refer to this [quickstart](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/fastchat_quickstart.html#).

For convenience, we have included a file `/llm/fastchat-examples/start-fastchat-service.sh` in the image.

You can modify this script to using fastchat with either `ipex_llm_worker` or `vllm_worker`.

#### vLLM serving engine

To run vLLM engine using `IPEX-LLM` as backend, you can refer to this [document](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md).

We have included multiple example files in `/llm/vllm-examples`:
1. `offline_inference.py`: Used for offline inference example
2. `benchmark_throughput.py`: Used for benchmarking throughput
3. `payload-1024.lua`: Used for testing request per second using 1k-128 request
4. `start-vllm-service.sh`: Used for template for starting vLLM service

##### Online benchmark throurgh api_server

We can benchmark the api_server to get an estimation about TPS (transactions per second).  To do so, you need to start the service first according to the instructions in this [section](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md#service).


In container, do the following:
1. modify the `/llm/vllm-examples/payload-1024.lua` so that the "model" attribute is correct.  By default, we use a prompt that is roughly 1024 token long, you can change it if needed.
2. Start the benchmark using `wrk` using the script below:

```bash
cd /llm/vllm-examples
# You can change -t and -c to control the concurrency.
# By default, we use 12 connections to benchmark the service.
wrk -t12 -c12 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h

```
#### Offline benchmark through benchmark_throughput.py

We have included the benchmark_throughput script provied by `vllm` in our image as `/llm/benchmark_throughput.py`.  To use the benchmark_throughput script, you will need to download the test dataset through:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

The full example looks like this:
```bash
cd /llm/vllm-examples

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

export MODEL="YOUR_MODEL"

# You can change load-in-low-bit from values in [sym_int4, fp8, fp16]

python3 /llm/vllm-examples/benchmark_throughput.py \
    --backend vllm \
    --dataset /llm/vllm-examples/ShareGPT_V3_unfiltered_cleaned_split.json \
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

> Note: you can adjust --load-in-low-bit to use other formats of low-bit quantization.


You can also adjust `--gpu-memory-utilization` rate using the below script to find the best performance using the following script:

```bash
#!/bin/bash

# Define the log directory
LOG_DIR="YOUR_LOG_DIR"
# Check if the log directory exists, if not, create it
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Define an array of model paths
MODELS=(
    "YOUR TESTED MODELS"
)

# Define an array of utilization rates
UTIL_RATES=(0.85 0.90 0.95)

# Loop over each model
for MODEL in "${MODELS[@]}"; do
    # Loop over each utilization rate
    for RATE in "${UTIL_RATES[@]}"; do
        # Extract a simple model name from the path for easier identification
        MODEL_NAME=$(basename "$MODEL")

        # Define the log file name based on the model and rate
        LOG_FILE="$LOG_DIR/${MODEL_NAME}_utilization_${RATE}.log"

        # Execute the command and redirect output to the log file
        # Sometimes you might need to set --max-model-len if memory is not enough
        # load-in-low-bit accepts inputs [sym_int4, fp8, fp16]
        python3 /llm/vllm-examples/benchmark_throughput.py \
            --backend vllm \
            --dataset /llm/vllm-examples/ShareGPT_V3_unfiltered_cleaned_split.json \
            --model $MODEL \
            --num-prompts 1000 \
            --seed 42 \
            --trust-remote-code \
            --enforce-eager \
            --dtype float16 \
            --load-in-low-bit sym_int4 \
            --device xpu \
            --gpu-memory-utilization $RATE &> "$LOG_FILE"
    done
done

# Inform the user that the script has completed its execution
echo "All benchmarks have been executed and logged."
```
