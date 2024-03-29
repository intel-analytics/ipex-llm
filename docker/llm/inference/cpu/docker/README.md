## Build/Use IPEX-LLM cpu image

### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT .
```


### Use the image for doing cpu inference


An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT

sudo docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```


After the container is booted, you could get into the container through `docker exec`.

To run inference using `IPEX-LLM` using cpu, you could refer to this [documentation](https://github.com/intel-analytics/IPEX-LLM/tree/main/python/llm#cpu-int4).

### Use chat.py

chat.py can be used to initiate a conversation with a specified model. The file is under directory '/llm'.

You can download models and bind the model directory from host machine to container when start a container.

Here is an example:
```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT
export MODEL_PATH=/home/llm/models

sudo docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        -v $MODEL_PATH:/llm/models/
        $DOCKER_IMAGE

```

After entering the container through `docker exec`, you can run chat.py by:
```bash
cd /llm
python chat.py --model-path YOUR_MODEL_PATH
```
In the example above, it can be:
```bash
cd /llm
python chat.py --model-path /llm/models/MODEL_NAME
```
