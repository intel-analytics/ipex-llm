# set -x
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export ENROLL_IMAGE_VERSION=latest
export ENROLL_IMAGE_NAME=bigdl-ppml-e2e-enroll
export JDK_URL=your_jdk_url

sudo docker build \
    --no-cache=true \
    --build-arg http_proxy=${HTTP_PROXY} \
    --build-arg https_proxy=${HTTPS_PROXY} \
    --build-arg JDK_URL=${JDK_URL} \
    -t $ENROLL_IMAGE_NAME:$ENROLL_IMAGE_VERSION -f ./Dockerfile .
