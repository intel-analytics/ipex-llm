#!/bin/bash

# KEYS_PATH means the absolute path to the keys folder
# ENCLAVE_KEY_PATH means the absolute path to the "enclave-key.pem" file
# LOCAL_IP means your local IP address.
export SSL_KEYS_PATH=YOUR_LOCAL_SSL_KEYS_FOLDER_PATH
export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=YOUR_DOCKER_IMAGE

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v $ENCLAVE_KEY_PATH:/root/.config/gramine/enclave-key.pem \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $SSL_KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=gramine-test \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash