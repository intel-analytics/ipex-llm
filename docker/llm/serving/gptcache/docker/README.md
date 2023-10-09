### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-llm-serving-gptcache-cpu:2.4.0-SNAPSHOT .
```


### Use the image for doing cpu inference


An example could be:
```bash
#/bin/bash
export DOCKER_IMAGE=intelanalytics/bigdl-llm-serving-gptcache-cpu:2.4.0-SNAPSHOT

sudo docker run -itd \
        --net=host \
        --cpuset-cpus="0-47" \
        --cpuset-mems="0" \
        --memory="32G" \
        --name=CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
```
