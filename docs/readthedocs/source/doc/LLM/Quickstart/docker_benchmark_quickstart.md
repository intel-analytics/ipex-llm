# Run Performance Benchmarking in Docker with IPEX-LLM on Intel GPU

Benchmarking IPEX-LLM on Intel GPUs within Docker can be efficiently achieved using provided benchmark scripts. Follow these steps to execute the process smoothly.

## Install Docker

1. Linux Installation
Follow the instructions in this [guide](https://www.docker.com/get-started/) to install Docker on Linux.

2. Windows Installation
For Windows installation, refer to this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/docker_windows_gpu.html#install-docker-on-windows).

## Prepare ipex-llm-xpu Docker Image

Run the following command to pull image from dockerhub:
```bash
docker pull intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
```

## Start ipex-llm-xpu Docker Container to Run Performance Benchmark

To map the xpu into the container, you need to specify --device=/dev/dri when booting the container.
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
export CONTAINER_NAME=my_container
export MODEL_PATH=/llm/models[change to your model path]

sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models \
        -e REPO_ID=<repo_id_value> \
        -e TEST_API=<test_api_value> \
        -e IN_OUT_PAIRS=<in_out_pairs_value> \
        -e DEVICE=<device_value> \
        $DOCKER_IMAGE benchmark.sh
```

Customize environment variables to specify:

- **REPO_ID:** Specify the model's name and organization, separated by commas if multiple values exist (e.g., "meta-llama/Llama-2-7b-chat-hf,THUDM/chatglm2-6b").
- **IN_OUT_PAIRS:** Define the combined input and output sequence lengths, separated by commas if multiple values exist (e.g., "32-32,1024-128").
- **TEST_API:** Utilize different test functions based on the machine, separated by commas if multiple values exist (e.g., "transformer_int4_gpu,transformer_int4_fp16_gp").
- **DEVICE:** Specify the type of device - Max, Flex, Arc.

## Result

After the benchmarking is completed, you can obtain a CSV result file under the current folder. You can mainly look at the results of columns `1st token avg latency (ms)` and `2+ avg latency (ms/token)` for the benchmark results. You can also check whether the column `actual input/output tokens` is consistent with the column `input/output tokens` and whether the parameters you specified in `config.yaml` have been successfully applied in the benchmarking.
