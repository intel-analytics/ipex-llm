export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  -t intelanalytics/ipex-llm-finetune-xpu:2.1.0-SNAPSHOT \
  -f ./Dockerfile .
