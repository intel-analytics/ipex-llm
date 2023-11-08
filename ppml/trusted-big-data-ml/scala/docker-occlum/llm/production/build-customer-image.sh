

export image=intelanalytics/bigdl-ppml-trusted-llm-fastchat-occlum-production
export TAG=2.5.0-SNAPSHOT-build
export image_customer=${image}-customer
pwd
docker build \
  --no-cache=true \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTP_PROXYS} \
  --build-arg no_proxy=${NO_PROXY} \
  --build-arg FINAL_NAME=${image}:${TAG} \
  -t ${image_customer}:${TAG} -f ./customer/Dockerfile .
