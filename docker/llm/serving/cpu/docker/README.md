## Build/Use IPEX-LLM-serving cpu image

### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT .
```

### Use the image for doing cpu serving


You could use the following bash script to start the container.  Please be noted that the CPU config is specified for Xeon CPUs, change it accordingly if you are not using a Xeon CPU.

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

After the container is booted, you could get into the container through `docker exec`.

#### FastChat serving engine
To run FastChat-serving using `IPEX-LLM` as backend, you can refer to this [document](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex_llm/serving/fastchat).

#### vLLM serving engine

To run vLLM engine using `IPEX-LLM` as backend, you can refer to this [document](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md).

We have included multiple example files in `/llm/`:
1. `vllm_offline_inference.py`: Used for vLLM offline inference example
2. `benchmark_vllm_throughput.py`: Used for benchmarking throughput
3. `payload-1024.lua`: Used for testing request per second using 1k-128 request
4. `start-vllm-service.sh`: Used for template for starting vLLM service

##### Online benchmark throurgh api_server

We can benchmark the api_server to get an estimation about TPS (transactions per second).  To do so, you need to start the service first according to the instructions in this [section](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md#service).


In container, do the following:
1. modify the `/llm/payload-1024.lua` so that the "model" attribute is correct.  By default, we use a prompt that is roughly 1024 token long, you can change it if needed.
2. Start the benchmark using `wrk` using the script below:

```bash
cd /llm
# You can change -t and -c to control the concurrency.
# By default, we use 12 connections to benchmark the service.
wrk -t4 -c4 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h

```
#### Offline benchmark through benchmark_vllm_throughput.py

We have included the benchmark_throughput script provied by `vllm` in our image as `/llm/benchmark_vllm_throughput.py`.  To use the benchmark_throughput script, you will need to download the test dataset through:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

The full example looks like this:
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