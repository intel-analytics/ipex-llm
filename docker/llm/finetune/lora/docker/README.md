## Prepare BigDL image for Lora Finetuning

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/bigdl-lora-finetuning:2.4.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg HTTP_PROXY=${HTTP_PROXY} \
  --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-lora-finetuning:2.4.0-SNAPSHOT \
  -f ./Dockerfile .
```
