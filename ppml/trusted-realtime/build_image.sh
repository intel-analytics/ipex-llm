#!/bin/bash
BASE_IMAGE_NAME=$BASE_IMAGE_NAME
BASE_IMAGE_TAG=$BASE_IMAGE_TAG
SGX_MEM_SIZE=$SGX_MEM_SIZE
SGX_LOG_LEVEL=$SGX_LOG_LEVEL
http_proxy=$http_proxy
https_proxy=$https_proxy

TARGET_IMAGE=$TARGET_IMAGE
TARGET_TAG=$TARGET_TAG

docker build   \
   --build-arg BASE_IMAGE_NAME=$BASE_IMAGE_NAME \
   --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG \
   --build-arg SGX_MEM_SIZE=$SGX_MEM_SIZE \
   --build-arg SGX_LOG_LEVEL=$SGX_LOG_LEVEL \
   --build-arg http_proxy=$http_proxy \
   --build-arg https_proxy=$https_proxy \
   --rm -t $TARGET_IMAGE:$TARGET_TAG -f Dockerfile .
