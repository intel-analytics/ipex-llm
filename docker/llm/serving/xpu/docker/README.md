# IPEX-LLM-serving XPU Image: Build and Usage Guide

This document outlines the steps to build and use the `IPEX-LLM-serving-xpu` Docker image, including inference, serving, and benchmarking functionalities for XPU.
---

## 1. Build the Image

To build the `IPEX-LLM-serving-xpu` Docker image, use the following command:

```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-serving-xpu:2.2.0-SNAPSHOT .
```

---

## 2. Using the Image for XPU Inference

To map the `XPU` into the container, you need to specify `--device=/dev/dri` when starting the container.

### Example:

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.2.0-SNAPSHOT

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```

Once the container is up and running, use `docker exec` to enter it.

To verify if the XPU device is successfully mapped into the container, run the following:

```bash
sycl-ls
```

For a machine with Arc A770, the output will be similar to:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```

For detailed instructions on running inference with `IPEX-LLM` on XPU, refer to this [documentation](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU).

---

## 3. Using the Image for XPU Serving

To run XPU serving, you need to map the XPU into the container by specifying `--device=/dev/dri` when booting the container.

### Example:

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```

After the container starts, access it using `docker exec`.

To verify that the device is correctly mapped, run:

```bash
sycl-ls
```

The output will be similar to the example in the inference section above.

Currently, the image supports two different serving engines: **FastChat** and **vLLM**.

### Serving Engines

#### 3.1 Lightweight Serving Engine

For running lightweight serving on Intel GPUs using `IPEX-LLM` as the backend, refer to the [Lightweight-Serving README](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Lightweight-Serving).

We have included a script `/llm/start-lightweight_serving-service` in the image. Make sure to install the correct `transformers` version before proceeding, like so:

```bash
pip install transformers==4.37.0
```

#### 3.2 Pipeline Parallel Serving Engine

To use the **Pipeline Parallel** serving engine with `IPEX-LLM` as the backend, refer to this [Pipeline-Parallel-FastAPI README](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/Pipeline-Parallel-FastAPI).

A convenience script `/llm/start-pp_serving-service.sh` is included in the image. Be sure to install the required version of `transformers`, like so:

```bash
pip install transformers==4.37.0
```

#### 3.3 vLLM Serving Engine

For running the **vLLM engine** with `IPEX-LLM` as the backend, refer to this [vLLM Docker Quickstart Guide](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/DockerGuides/vllm_docker_quickstart.md).

The following example files are available in `/llm/` within the container:

1. `vllm_offline_inference.py`: vLLM offline inference example
2. `benchmark_vllm_throughput.py`: Throughput benchmarking
3. `payload-1024.lua`: Request-per-second test (using 1k-128 request)
4. `start-vllm-service.sh`: Template for starting the vLLM service
5. `vllm_offline_inference_vision_language.py`: vLLM offline inference for vision-based models

---

## 4. Benchmarking

### 4.1 Online Benchmark through API Server

To benchmark the API server and estimate TPS (transactions per second), follow these steps:

1. Start the service as per the instructions in this [section](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/DockerGuides/vllm_docker_quickstart.md#Serving).
2. Run the benchmark using `vllm_online_benchmark.py`:

```bash
python vllm_online_benchmark.py $model_name $max_seqs $input_length $output_length
```

If `input_length` and `output_length` are not provided, the script defaults to values of 1024 and 512 tokens, respectively. The output will look something like:

```bash
model_name: Qwen1.5-14B-Chat
max_seq: 12
Warm Up: 100%|█████████████████████████████████████████████████████| 24/24 [01:36<00:00,  4.03s/req]
Benchmarking: 100%|████████████████████████████████████████████████| 60/60 [04:03<00:00,  4.05s/req]
Total time for 60 requests with 12 concurrent requests: xxx seconds.
Average response time: xxx
Token throughput: xxx

Average first token latency: xxx milliseconds.
P90 first token latency: xxx milliseconds.
P95 first token latency: xxx milliseconds.

Average next token latency: xxx milliseconds.
P90 next token latency: xxx milliseconds.
P95 next token latency: xxx milliseconds.
```

### 4.2 Online Benchmark with Multimodal Input

After starting the vLLM service, you can benchmark multimodal inputs using `vllm_online_benchmark_multimodal.py`:

```bash
export image_url="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
python vllm_online_benchmark_multimodal.py --model-name $model_name --image-url $image_url --prompt "What is in the image?" --port 8000
```

The `image_url` can be a local path (e.g., `/llm/xxx.jpg`) or an external URL (e.g., `"http://xxx.jpg`).

The output will be similar to the example in the API benchmarking section.

### 4.3 Online Benchmark through wrk

In the container, modify the `payload-1024.lua` to ensure the "model" attribute is correct. By default, it uses a prompt of about 1024 tokens.

Then, start the benchmark using `wrk`:

```bash
cd /llm
wrk -t12 -c12 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h
```

### 4.4 Offline Benchmark through `benchmark_vllm_throughput.py`

To use the `benchmark_vllm_throughput.py` script, first download the test dataset:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Then, run the benchmark:

```bash
cd /llm/

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

export MODEL="YOUR_MODEL"

python3 /llm/benchmark_vllm_throughput.py \
    --backend vllm \
    --dataset /llm/ShareGPT_V3_unfiltered_cleaned_split.json \
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

---