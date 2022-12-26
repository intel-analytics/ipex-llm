# set -x
export HTTP_PROXY=your_http_proxy
export HTTPS_PROXY=your_https_proxy
export IMAGE_NAME=your_image_name
export IMAGE_VERSION=your_image_version_tag
export JDK_URL=your_jdk_url

sudo docker build \
    --no-cache=true \
    --build-arg http_proxy=${HTTP_PROXY} \
    --build-arg https_proxy=${HTTPS_PROXY} \
    --build-arg JDK_URL=${JDK_URL} \
    -t $IMAGE_NAME:$IMAGE_VERSION -f ./Dockerfile .
