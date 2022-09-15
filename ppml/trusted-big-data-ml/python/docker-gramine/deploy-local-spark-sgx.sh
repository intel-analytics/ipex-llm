#!/bin/bash

# KEYS_PATH means the absolute path to the keys folder
# ENCLAVE_KEY_PATH means the absolute path to the "enclave-key.pem" file
# LOCAL_IP means your local IP address.
export SSL_KEYS_PATH=YOUR_LOCAL_SSL_KEYS_FOLDER_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=YOUR_DOCKER_IMAGE
export APP_ID=your_appid
export API_KEY=your_apikey

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
    -e ATTESTATION_ID=$APP_ID \
    -e ATTESTATION_KEY=$API_KEY \
    --name=gramine-test \
    -e LOCAL_IP=$LOCAL_IP \
    $DOCKER_IMAGE bash
