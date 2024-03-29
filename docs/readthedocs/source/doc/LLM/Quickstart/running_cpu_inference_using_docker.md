## IPEX-LLM inference on CPU quick start

This quickstart guide walks you through setting up and running large language model inference with `ipex-llm` using docker. 

### Prepare Docker Image

You can download directly from Dockerhub like(recommended):

```bash
docker pull intelanalytics/ipex-llm-cpu:2.5.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
cd 
docker images | grep intelanalytics/ipex-llm-cpu:2.5.0-SNAPSHOT
```

Or follow steps provided in [Build/Use IPEX-LLM cpu image](https://github.com/intel-analytics/ipex-llm/blob/main/docker/llm/README.md#ipex-llm-on-windows) to build the image from source:
```bash
docker build \
  --build-arg http_proxy$=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  --rm --no-cache -t intelanalytics/ipex-llm-cpu:2.5.0-SNAPSHOT .
```

Here we use Linux/MacOS as example, if you have a Windows OS, please follow [IPEX-LLM on Windows](https://github.com/intel-analytics/ipex-llm/blob/main/docker/llm/README.md#ipex-llm-on-windows) to prepare a IPEX-LLM inference image on CPU.

### Use the image for inference on CPU

Here, we use [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as example, please download it and start a docker container.

1. Download the model
``` python
from huggingface_hub import snapshot_download
repo_id="meta-llama/Llama-2-7b-chat-hf"
local_dir="./Mistral-7B-v0.1"
snapshot_download(repo_id=repo_id,
                  local_dir=local_dir,
                  local_dir_use_symlinks=False
                  )
```

2. start a docker container with files mounted like below:
```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export DOCKER_IMAGE=intelanalytics/ipex-llm-cpu:2.5.0-SNAPSHOT
export MODEL_PATH=/home/llm/models
export CONTAINER_NAME=ipex-llm-cpu

docker run -itd \
        --net=host \
        --privileged \
        --cpuset-cpus="0-7" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models/ \
        $DOCKER_IMAGE
```

### Start chatting

1. Enter the running container:

```bash
# Launch the Controller
docker exec -it ipex-llm-cpu bash
```

2. After entering the container through docker exec, you can run chat.py by:
``` bash
cd /llm/portable-zip
python chat.py --model-path /llm/models/Llama-2-7b-chat-hf
```

3. start chatting
You will see the example result
```
Human: how are you
BigDL-LLM: huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

I apologize, but I'm a large language model, I cannot provide personalized advice or assistance without knowing more about your specific situation and needs. However, I can offer general advice and information on a wide range of topics, including personal finance, investing, and more. Please feel free to ask me any questions you have, and I'll do my best to help.
```