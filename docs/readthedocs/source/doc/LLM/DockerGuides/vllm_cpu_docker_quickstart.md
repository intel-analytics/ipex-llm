# Serving using IPEX-LLM integrated vLLM on Intel CPU via docker

This guide demonstrates how to do LLM serving with `IPEX-LLM` integrated `vLLM` in Docker on Linux with Intel CPU.

## Install docker

Follow the instructions in this [guide](https://www.docker.com/get-started/) to install Docker on Linux.

## Pull the latest image

*Note: For running vLLM serving on Intel CPUs, you can currently use either the `intelanalytics/ipex-llm-serving-cpu:latest` or `intelanalytics/ipex-llm-serving-vllm-cpu:latest` Docker image.*

```bash
# This image will be updated every day
docker pull intelanalytics/ipex-llm-serving-cpu:latest
```

## Start Docker Container

To fully use your Intel CPU to run vLLM inference and serving, you should 
```
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:latest
export CONTAINER_NAME=ipex-llm-serving-cpu-container
sudo docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        -v /path/to/models:/llm/models \
        -e no_proxy=localhost,127.0.0.1 \
        --memory="64G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```

After the container is booted, you could get into the container through `docker exec`.

```bash
docker exec -it ipex-llm-serving-cpu-container /bin/bash
```

## Running vLLM serving with IPEX-LLM on Intel GPU in Docker

We have included multiple vLLM-related files in `/llm/`:
1. `vllm_offline_inference.py`: Used for vLLM offline inference example
2. `benchmark_vllm_throughput.py`: Used for benchmarking throughput
3. `payload-1024.lua`: Used for testing request per second using 1k-128 request
4. `start-vllm-service.sh`: Used for template for starting vLLM service

Before performing benchmark or starting the service, you can refer to this [section](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_cpu.html#environment-setup) to setup our recommended runtime configurations.

### Service

A script named `/llm/start-vllm-service.sh` have been included in the image for starting the service conveniently.

Modify the `model` and `served_model_name` in the script so that it fits your requirement. The `served_model_name` indicates the model name used in the API. 

Then start the service using `bash /llm/start-vllm-service.sh`, the following message should be print if the service started successfully.

If the service have booted successfully, you should see the output similar to the following figure:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/start-vllm-service.png" width=100%; />
</a>


#### Verify
After the service has been booted successfully, you can send a test request using `curl`. Here, `YOUR_MODEL` should be set equal to `served_model_name` in your booting script, e.g. `Qwen1.5`.

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

#### Tuning

You can tune the service using these four arguments:
- `--max-model-len`
- `--max-num-batched-token`
- `--max-num-seq`

You can refer to this [doc](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/vLLM_quickstart.html#service) for a detailed explaination on these parameters.

### Benchmark

#### Online benchmark throurgh api_server

We can benchmark the api_server to get an estimation about TPS (transactions per second).  To do so, you need to start the service first according to the instructions mentioned above.

Then in the container, do the following:
1. modify the `/llm/payload-1024.lua` so that the "model" attribute is correct.  By default, we use a prompt that is roughly 1024 token long, you can change it if needed.
2. Start the benchmark using `wrk` using the script below:

```bash
cd /llm
# warmup
wrk -t4 -c4 -d3m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h
# You can change -t and -c to control the concurrency.
# By default, we use 8 connections to benchmark the service.
wrk -t8 -c8 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h
```

#### Offline benchmark through benchmark_vllm_throughput.py

Please refer to this [section](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/vLLM_quickstart.html#performing-benchmark) on how to use `benchmark_vllm_throughput.py` for benchmarking.
