#!/bin/bash
export SSL_KEYS_PATH=path_of_your_ssl_keys_folder
export SSL_PASSWORD_PATH=path_of_your_ssl_password_folder
export CUSTOM_IMAGE_NAME=intelanalytics/bigdl-kms-reference:2.5.0-SNAPSHOT
export ROOT_KEY=your_256bit_base64_AES_key_string
export DATA_STORAGE_PATH=a_host_path_for_persistent_stoarge
export SGX_ENABLED=true # false for tdx docker user
export SERVER_PORT_NUM=9876

if [ "$SGX_ENABLED" = "true" ]; then
  sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $SSL_KEYS_PATH:/ppml/keys \
    -v $SSL_PASSWORD_PATH:/ppml/password \
    -v $DATA_STORAGE_PATH:/ppml/data \
    -e ROOT_KEY=${ROOT_KEY} \
    -e SGX_ENABLED=true \
    -e SERVER_PORT_NUM=${SERVER_PORT_NUM} \
    --name=bigdl-key-management-server \
    $CUSTOM_IMAGE_NAME bash
else
  sudo docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="0-5" \
      --oom-kill-disable \
      -v $SSL_KEYS_PATH:/ppml/keys \
      -v $SSL_PASSWORD_PATH:/ppml/password \
      -v $DATA_STORAGE_PATH:/ppml/data \
      -e ROOT_KEY=${ROOT_KEY} \
      -e SERVER_PORT_NUM=${SERVER_PORT_NUM} \
      --name=bigdl-key-management-server \
      $CUSTOM_IMAGE_NAME bash
fi
