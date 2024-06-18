## Finetune LLM on an Intel GPU via Docker

## Quick Start

### Install Docker

1. Linux Installation

    Follow the instructions in this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/docker_windows_gpu.html#linux) to install Docker on Linux.

2. Windows Installation

    For Windows installation, refer to this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/docker_windows_gpu.html#install-docker-desktop-for-windows).

#### Setting Docker on windows

Need to enable `--net=host`,follow [this guide](https://docs.docker.com/network/drivers/host/#docker-desktop) so that you can easily access the service running on the docker. The [v6.1x kernel version wsl]( https://learn.microsoft.com/en-us/community/content/wsl-user-msft-kernel-v6#1---building-the-microsoft-linux-kernel-v61x) is recommended to use.Otherwise, you may encounter the blocking issue before loading the model to GPU.

### Pull the latest image
```bash
# This image will be updated every day
docker pull intelanalytics/ipex-llm-xpu-finetune:latest
```

### Start Docker Container

```eval_rst
.. tabs::
   .. tab:: Linux

      To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. Select the device you are running(device type:(Max, Flex, Arc, iGPU)). And change the `/path/to/models` to mount the models. `bench_model` is used to benchmark quickly. If want to benchmark, make sure it on the `/path/to/models`

      .. code-block:: bash

        #/bin/bash
        export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu-finetune:latest
        export CONTAINER_NAME=ipex-llm-xpu-finetune-container
        sudo docker run -itd \
                --net=host \
                --device=/dev/dri \
                -v /path/to/models:/models \
                -e no_proxy=localhost,127.0.0.1 \
                --memory="32G" \
                --name=$CONTAINER_NAME \
                --shm-size="16g" \
                $DOCKER_IMAGE
   
   .. tab:: Windows

      To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. And change the `/path/to/models` to mount the models. Then add `--privileged` and map the `/usr/lib/wsl` to the docker.

      .. code-block:: bash

        #/bin/bash
        export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu-finetune:latest
        export CONTAINER_NAME=ipex-llm-xpu-finetune-container
        sudo docker run -itd \
                --net=host \
                --device=/dev/dri \
                --privileged \
                -v /path/to/models:/models \
                -v /usr/lib/wsl:/usr/lib/wsl \
                -e no_proxy=localhost,127.0.0.1 \
                --memory="32G" \
                --name=$CONTAINER_NAME \
                --shm-size="16g" \
                $DOCKER_IMAGE

```


After the container is booted, you could get into the container through `docker exec`.

```bash
docker exec -it ipex-llm-xpu-finetune-container /bin/bash
```

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```


### Example: Finetune Llama-3-8B with QLoRA (Peft)

Notice that the performance on windows wsl docker is a little slower than on windows host, ant it's caused by the implementation of wsl kernel.

```bash
docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=ipex-llm-finetune-xpu-container \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/alpaca-cleaned \
   --shm-size="16g" \
   intelanalytics/ipex-llm-finetune-xpu:latest
```

```bash
docker exec -it ipex-llm-finetune-xpu-container bash
```

```bash
cd /LLM-Finetuning/QLoRA/alpaca-qlora
```

```bash
python ./alpaca_qlora_finetuning.py \
    --base_model "/model" \
    --data_path "/data/alpaca-cleaned" \
    --output_dir "./ipex-llm-qlora-alpaca"
```


### Example: Finetune Llama-2-&B with QLoRA (TRL)

```bash
docker run -itd \
   --net=host \
   --device=/dev/dri \
   --memory="32G" \
   --name=ipex-llm-finetune-xpu-container \
   -e http_proxy=${HTTP_PROXY} \
   -e https_proxy=${HTTPS_PROXY} \
   -v $BASE_MODE_PATH:/model \
   -v $DATA_PATH:/data/alpaca-cleaned \
   --shm-size="16g" \
   intelanalytics/ipex-llm-finetune-xpu:latest
```

```bash
docker exec -it ipex-llm-finetune-xpu-container bash
```

```bash
cd /LLM-Finetuning/QLoRA/trl-example
```

```bash
python ./qlora_finetuning.py
    --repo-id-or-model-path /model/
    --dataset /data/alpaca-cleaned/
```

## Troubleshooting
