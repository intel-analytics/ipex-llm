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
