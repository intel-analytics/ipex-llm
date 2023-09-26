## Prepare BigDL image for Lora Finetuning

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/bigdl-llm-finetune-cpu:2.4.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/bigdl-llm-finetune-cpu:2.4.0-SNAPSHOT \
  -f ./Dockerfile .
```
