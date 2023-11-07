export image=intelanalytics/bigdl-ppml-trusted-llm-fastchat-occlum
export TAG=2.4.0
export image_production=${image}-production
pwd
docker build \
  --no-cache=true \
  --build-arg LLM_NAME=${image}:${TAG} \
  -t ${image_production}:${TAG} -f ./Dockerfile .
