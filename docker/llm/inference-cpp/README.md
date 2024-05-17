## Run llama.cpp/Ollama/Open-WebUI on an Intel GPU via Docker

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
docker pull intelanalytics/ipex-llm-inference-cpp-xpu:latest
```

### Start Docker Container

To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. Select the device you are running(device type:(Max, Flex, Arc, iGPU)). And change the `/path/to/models` to mount the models. `bench_model` is used to benchmark quickly. If want to benchmark, make sure it on the `/path/to/models`.

An Linux example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-inference-cpp-xpu:latest
export CONTAINER_NAME=ipex-llm-inference-cpp-xpu-container
sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        -v /path/to/models:/models \
        -e no_proxy=localhost,127.0.0.1 \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        -e bench_model="mistral-7b-v0.1.Q4_0.gguf" \
        -e DEVICE=Arc \
        --shm-size="16g" \
        $DOCKER_IMAGE
```

An Windows example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-inference-cpp-xpu:latest
export CONTAINER_NAME=ipex-llm-inference-cpp-xpu-container
sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --privileged \
        -v /path/to/models:/models \
        -v /usr/lib/wsl:/usr/lib/wsl \
        -e no_proxy=localhost,127.0.0.1 \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        -e bench_model="mistral-7b-v0.1.Q4_0.gguf" \
        -e DEVICE=Arc \
        --shm-size="16g" \
        $DOCKER_IMAGE
```


After the container is booted, you could get into the container through `docker exec`.

```bash
docker exec -it ipex-llm-inference-cpp-xpu-container /bin/bash
```

To verify the device is successfully mapped into the container, run `sycl-ls` to check the result. In a machine with Arc A770, the sampled output is:

```bash
root@arda-arc12:/# sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2023.16.7.0.21_160000]
[opencl:cpu:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i9-13900K 3.0 [2023.16.7.0.21_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics 3.0 [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26241]
```


### Quick benchmark for llama.cpp

Notice that the performance on windows wsl docker is a little slower than on windows host, ant it's caused by the implementation of wsl kernel.

```bash
bash /llm/scripts/benchmark_llama-cpp.sh

# benchmark results
llama_print_timings:        load time =    xxx ms
llama_print_timings:      sample time =       xxx ms /    xxx runs   (    xxx ms per token, xxx tokens per second)
llama_print_timings: prompt eval time =     xxx ms /    xxx tokens (    xxx ms per token,   xxx tokens per second)
llama_print_timings:        eval time =     xxx ms /    128 runs   (   xxx ms per token,    xxx tokens per second)
llama_print_timings:       total time =     xxx ms /    xxx tokens
```


### Running llama.cpp inference with IPEX-LLM on Intel GPU

```bash
cd /llm/scripts/
# set the recommended Env
source ipex-llm-init --gpu --device $DEVICE
# mount models and change the model_path in `start-llama-cpp.sh`
bash start-llama-cpp.sh
```

Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html) for more details.


### Running Ollama serving with IPEX-LLM on Intel GPU

Running the ollama on the background, you can see the ollama.log in `/root/ollama/ollama.log`
```bash
cd /llm/scripts/
# set the recommended Env
source ipex-llm-init --gpu --device $DEVICE
bash start-ollama.sh # ctrl+c to exit
```

#### Run Ollama models (interactive)

```bash
cd /llm/ollama
# create a file named Modelfile
FROM /models/mistral-7b-v0.1.Q4_0.gguf
TEMPLATE [INST] {{ .Prompt }} [/INST]
PARAMETER num_predict 64

# create example and run it on console
./ollama create example -f Modelfile
./ollama run example
```

#### Pull models from ollama to serve

```bash
cd /llm/ollama
./ollama pull llama2
```

Use the Curl to Test:
```bash
curl http://localhost:11434/api/generate -d '
{ 
   "model": "llama2", 
   "prompt": "What is AI?", 
   "stream": false
}'
```

Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html#pull-model) for more details.


### Running Open WebUI with Intel GPU

Start the ollama and load the model first, then use the open-webui to chat.
If you have difficulty accessing the huggingface repositories, you may use a mirror, e.g. add export HF_ENDPOINT=https://hf-mirror.com before running bash start.sh.
```bash
cd /llm/scripts/
bash start-open-webui.sh
# INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

For how to log-in or other guide, Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/open_webui_with_ollama_quickstart.html) for more details.
