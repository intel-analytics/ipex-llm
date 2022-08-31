#!/bin/bash

export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export DATA_PATH=YOUR_LOCAL_DATA_PATH
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-fl-graphene:2.1.0-SNAPSHOT

sudo docker rm -f fl-server

sudo docker run -it \
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
    --name=fl-server \
    -e SGX_MEM_SIZE=32G \
    -e SGX_LOG_LEVEL=error \
    $DOCKER_IMAGE bash /ppml/trusted-big-data-ml/runFlServer.sh
