#!/bin/bash
export SSL_KEYS_PATH=path_of_your_ssl_keys_folder
export SSL_PASSWORD_PATH=path_of_your_ssl_password_folder
export IMAGE_NAME=intelanalytics/easy-kms:2.3.0-SNAPSHOT
export ROOT_KEY=your_256bit_base64_AES_key_string

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    -v $SSL_KEYS_PATH:/ppml/keys \
    -v $SSL_PASSWORD_PATH:/ppml/password \
    -e ROOT_KEY=${ROOT_KEY} \
    --name=easy-key-management-server \
    $IMAGE_NAME bash
