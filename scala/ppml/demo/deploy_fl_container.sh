#!/bin/bash

export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export DATA_PATH=YOUR_LOCAL_DATA_PATH
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=iintelanalytics/bigdl-ppml-trusted-fl-graphene:0.14.0-SNAPSHOT

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-19" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=flDemo \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=32G \
    -e SGX_LOG_LEVEL=error \
    $DOCKER_IMAGE bash
