## BigDL-LLM inference on GPU quick start

This quickstart guide walks you through setting up and running large language model inference with `bigdl-llm` using docker on Intel GPU. 

### Prepare Docker Image

You can download directly from Dockerhub like(recommended):

```bash
docker pull intelanalytics/bigdl-llm-serving-xpu:2.5.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
docker images | grep intelanalytics/bigdl-llm-serving-xpu
```

Or follow steps provided in [Build/Use BigDL-LLM cpu image](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/serving/cpu/docker) to build the image from source:
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-llm-serving-xpu:2.5.0-SNAPSHOT .
```

Here we use Linux/MacOS as example, if you have a Windows OS, please follow [BigDL-LLM on Windows](https://github.com/intel-analytics/BigDL/blob/main/docker/llm/README.md#bigdl-llm-on-windows) to prepare a BigDL-LLM inference image on CPU.

### Use the image for inference on CPU

Here, we use [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) as example, please download it and start a docker container with files mounted like below:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-xpu:2.5.0-SNAPSHOT
export MODEL_PATH=/home/llm/models
export CONTAINER_NAME=bigdl-llm-inference-gpu
export SERVICE_MODEL_PATH=/llm/models/Mistral-7B-v0.1 [a specified model path for running service]

docker run -itd \
    --net=host \
    --device=/dev/dri \
    --memory="32G" \
    --name=$CONTAINER_NAME \
    --shm-size="16g" \
    -v $MODEL_PATH:/llm/models \
    -e SERVICE_MODEL_PATH=$SERVICE_MODEL_PATH \
    $DOCKER_IMAGE --service-model-path $SERVICE_MODEL_PATH
```

### Start the inference service with Web UI

Enter the running container and start a service:

```bash
# Launch the Controller
docker exec -itd bigdl-llm-inference-gpu python3 -m fastchat.serve.controller
# Launch the model worker(s)
docker exec -itd bigdl-llm-inference-gpu python3 -m bigdl.llm.serving.model_worker --model-path /llm/models/Mistral-7B-v0.1  --device cpu
# Launch the Gradio web server
docker exec -itd bigdl-llm-inference-gpu python3 -m fastchat.serve.gradio_web_server --port 7860 
```
This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI with BigDL-LLM as the backend. You can open your browser and chat with a Mistral-7B-v0.1 now.

### Chat with a model

See the results on your web browser http://your-ip:7860

