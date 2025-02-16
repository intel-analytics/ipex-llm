# IPEX-LLM-Serving CPU Image: Build and Usage Guide

This document provides instructions for building and using the `IPEX-LLM-serving` CPU Docker image, including model inference, serving, and benchmarking functionalities.


---

## 1. Build the Image

To build the `ipex-llm-serving-cpu` Docker image, run the following command:

```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT .
```

---

## 2. Using `chat.py` for Inference

The `chat.py` script is used for model inference. It is located under the `/llm` directory inside the container.

### Steps:
1. **Download the model** to your host machine and bind the model directory to the container when starting it.

#### Example command to run the container:
```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT
export MODEL_PATH=/home/llm/models

sudo docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models/ \
        $DOCKER_IMAGE
```

2. **Run `chat.py` for inference** inside the container:
After entering the container, run the following command to start the inference:

```bash
cd /llm
python chat.py --model-path /llm/models/MODEL_NAME
```

Replace `MODEL_NAME` with the name of your model.

---

## 3. CPU Serving with `IPEX-LLM`

To run CPU-side serving with `IPEX-LLM`, follow these steps:

### Start the container:
Use the following bash script to start the container. Please be noted that the CPU config is specified for Xeon CPUs, change it accordingly if you are not using a Xeon CPU.

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT

sudo docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```

Once the container is started, you can access it using `docker exec`.

---

## 4. Serving with FastChat Engine

To run FastChat-serving using `IPEX-LLM` as backend, you can refer to this [document](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex_llm/serving/fastchat).

---

## 5. Serving with vLLM Engine

To use **vLLM** with `IPEX-LLM` as the backend, refer to the [vLLM Serving Guide](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md).

We have included the following example files in the `/llm/` directory inside the container:

- `vllm_offline_inference.py`: Used for vLLM offline inference example.
- `benchmark_vllm_throughput.py`: Used for throughput benchmarking.
- `payload-1024.lua`: Used for testing requests per second with a 1k-128 request pattern.
- `start-vllm-service.sh`: Template script for starting the vLLM service.

---

## 6. Benchmarks

### 6.1 Online Benchmark through API Server

To benchmark the API Server and estimate transactions per second (TPS). To do so, you need to start the service first according to the instructions in this [section](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md#service). Then follow these steps:

1. **Modify the `payload-1024.lua` file** to ensure that the "model" attribute is correctly set. By default, it uses a prompt that is approximately 1024 tokens long, which can be adjusted if needed.
2. **Run the benchmark using `wrk`**:
   Execute the following script to run the benchmark:

```bash
cd /llm
# You can adjust -t and -c to control concurrency.
# By default, we use 12 connections for the benchmark.
wrk -t4 -c4 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h
```

### 6.2 Offline Benchmark through `benchmark_vllm_throughput.py`

We have included the vLLM throughput benchmark script in `/llm/benchmark_vllm_throughput.py`. To use the script, download the test dataset first:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Then, run the benchmark script:

```bash
cd /llm/

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

export MODEL="YOUR_MODEL"

# You can change load-in-low-bit from values in [sym_int4, fp8, fp16]
python3 /llm/benchmark_vllm_throughput.py \
    --backend vllm \
    --dataset /llm/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model $MODEL \
    --num-prompts 1000 \
    --seed 42 \
    --trust-remote-code \
    --enforce-eager \
    --dtype bfloat16 \
    --device cpu \
    --load-in-low-bit sym_int4
```

---
