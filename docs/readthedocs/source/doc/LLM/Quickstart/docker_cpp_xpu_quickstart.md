## Run llama.cpp/Ollama/open-webui with Docker on Intel GPU

## Quick Start

### Install Docker

1. Linux Installation

    Follow the instructions in this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/docker_windows_gpu.html#linux) to install Docker on Linux.

2. Windows Installation

    For Windows installation, refer to this [guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/DockerGuides/docker_windows_gpu.html#install-docker-desktop-for-windows).

#### Setting Docker on windows
If you want to run this image on windows, please refer to (this document)[https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/docker_windows_gpu.html#install-docker-on-windows] to set up Docker on windows. And run below steps on wls ubuntu. And you need to enable `--net=host`,follow [this guide](https://docs.docker.com/network/drivers/host/#docker-desktop) so that you can easily access the service running on the docker. The [v6.1x kernel version wsl]( https://learn.microsoft.com/en-us/community/content/wsl-user-msft-kernel-v6#1---building-the-microsoft-linux-kernel-v61x) is recommended to use.Otherwise, you may encounter the blocking issue before loading the model to GPU.

### Pull the latest image
```bash
# This image will be updated every day
docker pull intelanalytics/ipex-llm-inference-cpp-xpu:latest
```

### Start Docker Container

```eval_rst
.. tabs::
   .. tab:: Linux

      To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. Select the device you are running(device type:(Max, Flex, Arc, iGPU)). And change the `/path/to/models` to mount the models. `bench_model` is used to benchmark quickly. If want to benchmark, make sure it on the `/path/to/models`

      .. code-block:: bash

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
   
   .. tab:: Windows

      To map the `xpu` into the container, you need to specify `--device=/dev/dri` when booting the container. And change the `/path/to/models` to mount the models. Then add `--privileged` and map the `/usr/lib/wsl` to the docker.

      .. code-block:: bash

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
```

The benchmark will run three times to warm up to get the accurate results, and the example output is like:
```bash
llama_print_timings:        load time =    xxx ms
llama_print_timings:      sample time =       xxx ms /    128 runs   (    xxx ms per token, xxx tokens per second)
llama_print_timings: prompt eval time =     xxx ms /    xxx tokens (    xxx ms per token,   xxx tokens per second)
llama_print_timings:        eval time =     xxx ms /    127 runs   (   xxx ms per token,    xxx tokens per second)
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

The example output is like:
```bash
llama_print_timings:        load time =    xxx ms
llama_print_timings:      sample time =       xxx ms /    32 runs   (    xxx ms per token, xxx tokens per second)
llama_print_timings: prompt eval time =     xxx ms /    xxx tokens (    xxx ms per token,   xxx tokens per second)
llama_print_timings:        eval time =     xxx ms /    31 runs   (   xxx ms per token,    xxx tokens per second)
llama_print_timings:       total time =     xxx ms /    xxx tokens
```

Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html) for more details.


### Running Ollama serving with IPEX-LLM on Intel GPU

Running the ollama on the background, you can see the ollama.log in `/root/ollama/ollama.log`
```bash
cd /llm/scripts/
# set the recommended Env
source ipex-llm-init --gpu --device $DEVICE
bash start-ollama.sh # ctrl+c to exit, and the ollama serve will run on the background
```

Sample output:
```bash
time=2024-05-16T10:45:33.536+08:00 level=INFO source=images.go:697 msg="total blobs: 0"
time=2024-05-16T10:45:33.536+08:00 level=INFO source=images.go:704 msg="total unused blobs removed: 0"
time=2024-05-16T10:45:33.536+08:00 level=INFO source=routes.go:1044 msg="Listening on 127.0.0.1:11434 (version 0.0.0)"
time=2024-05-16T10:45:33.537+08:00 level=INFO source=payload.go:30 msg="extracting embedded files" dir=/tmp/ollama751325299/runners
time=2024-05-16T10:45:33.565+08:00 level=INFO source=payload.go:44 msg="Dynamic LLM libraries [cpu cpu_avx cpu_avx2]"
time=2024-05-16T10:45:33.565+08:00 level=INFO source=gpu.go:122 msg="Detecting GPUs"
time=2024-05-16T10:45:33.566+08:00 level=INFO source=cpu_common.go:11 msg="CPU has AVX2"
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

An example process of interacting with model with `ollama run example` looks like the following:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" width=100%; />
</a>


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

Sample output:
```bash
{"model":"llama2","created_at":"2024-05-16T02:52:18.972296097Z","response":"\nArtificial intelligence (AI) refers to the development of computer systems that can perform tasks that typically require human intelligence, such as learning, problem-solving, and decision-making. AI systems use algorithms and data to mimic human behavior and perform tasks such as:\n\n1. Image recognition: AI can identify objects in images and classify them into different categories.\n2. Natural Language Processing (NLP): AI can understand and generate human language, allowing it to interact with humans through voice assistants or chatbots.\n3. Predictive analytics: AI can analyze data to make predictions about future events, such as stock prices or weather patterns.\n4. Robotics: AI can control robots that perform tasks such as assembly, maintenance, and logistics.\n5. Recommendation systems: AI can suggest products or services based on a user's past behavior or preferences.\n6. Autonomous vehicles: AI can control self-driving cars that can navigate through roads and traffic without human intervention.\n7. Fraud detection: AI can identify and flag fraudulent transactions, such as credit card purchases or insurance claims.\n8. Personalized medicine: AI can analyze genetic data to provide personalized medical recommendations, such as drug dosages or treatment plans.\n9. Virtual assistants: AI can interact with users through voice or text interfaces, providing information or completing tasks.\n10. Sentiment analysis: AI can analyze text or speech to determine the sentiment or emotional tone of a message.\n\nThese are just a few examples of what AI can do. As the technology continues to evolve, we can expect to see even more innovative applications of AI in various industries and aspects of our lives.","done":true,"context":[xxx,xxx],"total_duration":12831317190,"load_duration":6453932096,"prompt_eval_count":25,"prompt_eval_duration":254970000,"eval_count":390,"eval_duration":6079077000}
```


Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html#pull-model) for more details.


### Running Open WebUI with Intel GPU

Start the ollama and load the model first, then use the open-webui to chat.
If you have difficulty accessing the huggingface repositories, you may use a mirror, e.g. add export HF_ENDPOINT=https://hf-mirror.com before running bash start.sh.
```bash
cd /llm/scripts/
bash start-open-webui.sh
```

Sample output:
```bash
INFO:     Started server process [1055]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_signup.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_signup.png" width="100%" />
</a>

For how to log-in or other guide, Please refer to this [documentation](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/open_webui_with_ollama_quickstart.html) for more details.
