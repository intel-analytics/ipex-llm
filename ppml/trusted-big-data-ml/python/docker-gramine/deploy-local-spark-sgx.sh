#!/bin/bash

export KEYS_PATH=YOUR_LOCAL_SPARK_SSL_KEYS_FOLDER_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export CUSTOM_IMAGE=YOUR_SELF_BUILD_CUSTOM_IMAGE

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $SSL_KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=gramine-test \
    -e LOCAL_IP=$LOCAL_IP \
    $CUSTOM_IMAGE bash
