## BigDL-LLM inference on CPU quick start

This quickstart guide walks you through setting up and running large language model inference with `bigdl-llm` using docker. 

### Step 1. Prepare Docker Image

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics//bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
docker images | grep intelanalytics//bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
```

Or follow steps provided in [Build/Use BigDL-LLM cpu image](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/serving/cpu/docker) to build the image from source:
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics//bigdl-llm-serving-cpu:2.5.0-SNAPSHOT .
```

Here we use Linux/MacOS as example, if you have a Windows OS, please follow [BigDL-LLM on Windows](https://github.com/intel-analytics/BigDL/blob/main/docker/llm/README.md#bigdl-llm-on-windows) to prepare a BigDL-LLM inference on CPU image.

### Step 2. Use the image for inference on CPU

Here, we use [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) as example, please download it and start a docker container with files mounted like below:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export DOCKER_IMAGE=intelanalytics//bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
export MODEL_PATH=/home/llm/models
export CONTAINER_NAME=bigdl-llm-inference-cpu

docker run -itd \
        --net=host \
        --privileged \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models/ \
        $DOCKER_IMAGE
```

### Step 3. See the results (Local Mode)

Enter the running container:

```bash
docker exec -itd bigdl-llm-inference-cpu python3 -m fastchat.serve.controller
docker exec -itd bigdl-llm-inference-cpu python3 -m bigdl.llm.serving.model_worker --model-path /llm/models/Mistral-7B-v0.1  --device cpu
docker exec -itd bigdl-llm-inference-cpu python3 -m fastchat.serve.gradio_web_server
```

See the results on your web browser http://10.112.242.181:7860

