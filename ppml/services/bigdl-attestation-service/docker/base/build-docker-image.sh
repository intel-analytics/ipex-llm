# set -x
export BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-base
export BASE_IMAGE_TAG=2.5.0-SNAPSHOT
export IMAGE_NAME=intelanalytics/bigdl-attestation-service-base
export IMAGE_VERSION=2.5.0-SNAPSHOT

sudo docker build \
    --no-cache=true \
    --build-arg BASE_IMAGE_NAME=${BASE_IMAGE_NAME} \
    --build-arg BASE_IMAGE_TAG=${BASE_IMAGE_TAG} \
    -t $IMAGE_NAME:$IMAGE_VERSION -f ./Dockerfile .
