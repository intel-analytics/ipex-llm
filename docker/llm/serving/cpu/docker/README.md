## Build/Use BigDL-LLM-serving cpu image

### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT .
```

### Use the image for doing cpu serving


You could use the following bash script to start the container.  Please be noted that the CPU config is specified for Xeon CPUs, change it accordingly if you are not using a Xeon CPU.

```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT

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

To run model-serving using `BigDL-LLM` as backend, you can refer to this [document](https://github.com/intel-analytics/BigDL/tree/main/python/llm/src/bigdl/llm/serving).
Also you can set environment variables and start arguments while running a container to get serving started initially. You may need to boot several containers to support. One controller container and at least one worker container are needed. The api server address(host and port) and controller address are set in controller container, and you need to set the same controller address as above, model path on your machine and worker address in worker container.

To start a controller container:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
controller_host=localhost
controller_port=23000
api_host=localhost
api_port=8000
sudo docker run -itd \
        --net=host \
	--privileged \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="64G" \
        --name=serving-cpu-controller \
        --shm-size="16g" \
	-e ENABLE_PERF_OUTPUT="true" \
        -e CONTROLLER_HOST=$controller_host \
        -e CONTROLLER_PORT=$controller_port \
        -e API_HOST=$api_host \
        -e API_PORT=$api_port \
        $DOCKER_IMAGE -m controller
```
To start a worker container:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-cpu:2.5.0-SNAPSHOT
export MODEL_PATH=YOUR_MODEL_PATH
controller_host=localhost
controller_port=23000
worker_host=localhost
worker_port=23001
sudo docker run -itd \
        --net=host \
	--privileged \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="64G" \
        --name="serving-cpu-worker" \
        --shm-size="16g" \
	-e ENABLE_PERF_OUTPUT="true" \
        -e CONTROLLER_HOST=$controller_host \
        -e CONTROLLER_PORT=$controller_port \
        -e WORKER_HOST=$worker_host \
        -e WORKER_PORT=$worker_port \
        -e OMP_NUM_THREADS=48 \
        -e MODEL_PATH=/llm/models/Llama-2-7b-chat-hf \
	-v $MODEL_PATH:/llm/models/ \
        $DOCKER_IMAGE -m worker -w vllm_worker # use -w model_worker if vllm worker is not needed
```

Then you can use `curl` for testing, an example could be:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "model": "YOUR_MODEL_NAME",
    "prompt": "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
    "n": 1,
    "best_of": 1,
    "use_beam_search": false,
    "stream": false
}' http://localhost:8000/v1/completions
```