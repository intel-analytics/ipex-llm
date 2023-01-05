# set -x
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export IMAGE_NAME=intelanalytics/bigdl-attestation-service
export IMAGE_VERSION=2.2.0-SNAPSHOT
export JDK_URL=your_jdk_url

sudo docker build \
    --no-cache=true \
    --build-arg http_proxy=${HTTP_PROXY} \
    --build-arg https_proxy=${HTTPS_PROXY} \
    --build-arg JDK_URL=${JDK_URL} \
    -t $IMAGE_NAME:$IMAGE_VERSION -f ./Dockerfile .
