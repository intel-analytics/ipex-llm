## IPEX-LLM inference on GPU quick start

This quickstart guide walks you through setting up and running large language model inference with `ipex-llm` using docker. 

### Docker Installation Instructions
**For New Users**:
- Begin by visiting the official Docker Get Started page for a comprehensive introduction and installation guide.

### Prepare Docker Image

You can download directly from Dockerhub like(recommended):

```bash
docker pull intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
```
to check if the image is successfully downloaded, you can use:

```bash
docker images | grep intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
```

### Download the model 
Here, we use [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as example to show LLM inference. Create a ``download.py`` and insert the code snippet below to dwonload the model from huggingface. 

``` python
from huggingface_hub import snapshot_download
repo_id="meta-llama/Llama-2-7b-chat-hf"
local_dir="/home/llm/models/Llama-2-7b-chat-hf"
snapshot_download(repo_id=repo_id,
                  local_dir=local_dir,
                  local_dir_use_symlinks=False
                  )
```

Then use the script to download the model to local directory of ``/home/llm/models/Llama-2-7b-chat-hf``. 
``` bash
pip install huggingface_hub
python download.py
```

### Start a docker container and run inference

Now you can start a docker container and use example of chat.py to run inference with 
``Llama-2-7b-chat-hf`` model.

#### Start a docker container
Set up your proxy and start the container with a name of ``ipex-llm-xpu`` by calling ``docker run``.

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:2.1.0-SNAPSHOT
export MODEL_PATH=/home/llm/models/
export CONTAINER_NAME=ipex-llm-xpu

docker run -itd \
    --net=host \
    --device=/dev/dri \
    --memory="32G" \
    --name=$CONTAINER_NAME \
    --shm-size="16g" \
    -v $MODEL_PATH:/llm/models \
    $DOCKER_IMAGE
```

#### Enter the running container
After the container is booted, you could get into the container through docker exec.

```bash
# Launch the Controller
docker exec -it ipex-llm-xpu bash
```

#### run chat.py example
After entering the container through docker exec, you can run chat.py by:

``` bash
cd /llm
python chat.py --model-path /llm/models/Llama-2-7b-chat-hf
```

#### Chat with the model and see the responses 

You will see the example result after you type "what is AI?".
```
Human: what is AI?
BigDL-LLM: Hello there! I'm here to help you understand AI. AI stands for Artificial Intelligence, which refers to the development of computer systems that can perform tasks that typically require human intelligence, such as learning, problem-solving, decision-making, and communication.
```