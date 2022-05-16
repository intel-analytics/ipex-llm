#!/bin/bash

export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export DATA_PATH=YOUR_LOCAL_DATA_PATH
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-fl-graphene:2.1.0-SNAPSHOT

arg=$1
case "$arg" in
    hfl1)
        export pod_name=hfl-client1
        ;;
    hfl2)
        export pod_name=hfl-client2
        ;;
    vfl1)
        export pod_name=vfl-client1
        ;;
    vfl2)
        export pod_name=Vfl-client2
        ;;
esac

sudo docker rm -f $pod_name
sudo docker run -it \
    --privileged \
    --net=host \
    --name=$pod_name \
    --cpuset-cpus="20-39" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    -e SGX_MEM_SIZE=32G \
    -e SGX_LOG_LEVEL=error \
    $DOCKER_IMAGE bash $1
