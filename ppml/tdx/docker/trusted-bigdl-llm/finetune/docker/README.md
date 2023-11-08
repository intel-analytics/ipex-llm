## Prepare BigDL image for Lora Finetuning

You can download directly from Dockerhub like:

```bash
docker pull intelanalytics/trusted-bigdl-llm-finetune-tdx:2.5.0-SNAPSHOT
```

Or build the image from source:

```bash
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy

export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export BIGDL_LLM_IMAGE_NAME=intelanalytics/bigdl-llm-finetune-cpu # or your custom native llm image name
export BIGDL_LLM_IMAGE_TAG=2.5.0-SNAPSHOT # or your custom image tag

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  --build-arg BIGDL_LLM_IMAGE_NAME=${BIGDL_LLM_IMAGE_NAME} \
  --build-arg BIGDL_LLM_IMAGE_TAG=${BIGDL_LLM_IMAGE_TAG} \
  -t intelanalytics/trusted-bigdl-llm-finetune-tdx:2.5.0-SNAPSHOT \
  -f ./Dockerfile .
```
