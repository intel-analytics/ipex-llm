export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port

export image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production
export TAG=2.5.0-SNAPSHOT-build
export image_customer=${image}-customer
pwd
docker build \
  --no-cache=true \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  --build-arg HTTP_PROXY_HOST=${HTTP_PROXY_HOST_2} \
  --build-arg HTTP_PROXY_PORT=${HTTP_PROXY_PORT_2} \
  --build-arg HTTPS_PROXY_HOST=${HTTP_PROXY_HOST_2} \
  --build-arg HTTPS_PROXY_PORT=${HTTP_PROXY_PORT_3} \
  --build-arg no_proxy=${NO_PROXY} \
  --build-arg FINAL_NAME=${image}:${TAG} \
  --build-arg SPARK_JAR_REPO_URL=${SPARK_JAR_REPO_URL} \
  -t ${image_customer}:${TAG} -f ./production/customer/Dockerfile .
