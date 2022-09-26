#!/bin/bash

export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0

sudo docker pull $DOCKER_IMAGE

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash
