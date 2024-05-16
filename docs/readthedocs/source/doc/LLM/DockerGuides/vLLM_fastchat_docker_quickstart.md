# Serving using IPEX-LLM and vLLM on Intel GPUs via docker

This guide demonstrates how to serve using `IPEX-LLM` with `vLLM` in Docker on Linux with Intel GPUs.

## Install docker

Follow the instructions in this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/docker_windows_gpu.html#linux) to install Docker on Linux.

## Pull the latest image

```bash
# This image will be updated every day
docker pull intelanalytics/ipex-llm-serving-xpu:latest
```

## Start Docker Container

 To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. Change the `/path/to/models` to mount the models. 

      .. code-block:: bash

        #/bin/bash
        export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
        export CONTAINER_NAME=ipex-llm-serving-xpu-container
        sudo docker run -itd \
                --net=host \
                --device=/dev/dri \
                -v /path/to/models:/llm/models \
                -e no_proxy=localhost,127.0.0.1 \
                --memory="32G" \
                --name=$CONTAINER_NAME \
                --shm-size="16g" \
                $DOCKER_IMAGE


After the container is booted, you could get into the container through `docker exec`.

```bash
docker exec -it ipex-llm-serving-xpu-container /bin/bash
```


To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```


## Running FastChat serving with IPEX-LLM on Intel GPU in Docker

### Using provided script

After `exec` into the container, you will be in the workdir `/llm`.  To start the FastChat service, you can use our provided script `/llm/start-fastchat-service.sh`.

```bash
# Only the MODEL_PATH needs to be set, other parameters have default values
export MODEL_PATH=YOUR_SELECTED_MODEL_PATH
export LOW_BIT_FORMAT=sym_int4
export CONTROLLER_HOST=localhost
export CONTROLLER_PORT=21001
export WORKER_HOST=localhost
export WORKER_PORT=21002
export API_HOST=localhost
export API_PORT=8000

# Use the default model_worker
bash /llm/start-fastchat-service.sh -w model_worker

# If you want to use the vllm_worker, then use the following command:
# bash /llm/start-fastchat-service.sh -w vllm_worker
```

If everything goes smoothly, the result should be similar to the following figure:

TODO: add a figure about starting FastChat

### Start the service manually

If the provided script cannot fulfill the service requirements or the `openai_api_server` need to be used, then this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/fastchat_quickstart.html#start-the-service) can be used for reference.


## Running vLLM serving with IPEX-LLM on Intel GPU in docker

### Using provided script