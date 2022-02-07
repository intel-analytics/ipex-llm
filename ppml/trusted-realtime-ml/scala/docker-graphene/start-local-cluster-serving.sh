#!/bin/bash

export ENCLAVE_KEY_PATH=the_dir_of_your_enclave_key
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export SECURE_PASSWORD_PATH=the_dir_path_of_your_prepared_password
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-30" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v $KEYS_PATH:/ppml/trusted-realtime-ml/redis/work/keys \
    -v $KEYS_PATH:/ppml/trusted-realtime-ml/java/work/keys \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/redis/work/password \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/java/work/password \
    --name=trusted-cluster-serving-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e CORE_NUM=30 \
    intelanalytics/bigdl-ppml-trusted-realtime-ml-scala-graphene:0.14.0-SNAPSHOT \
    bash  -c "cd /ppml/trusted-realtime-ml/ && ./start-all.sh && tail -f /dev/null"

sudo docker exec -i trusted-cluster-serving-local bash -c "mkdir /dev/sgx && \
    ln -s /dev/sgx_enclave /dev/sgx/enclave && \
    ln -s /dev/sgx_provision /dev/sgx/provision"
