#!/bin/bash
set -x

source ./environment.sh
mkdir ./zoo
mkdir ./pyzoo
export ZOO_PATH=zoo
export PYZOO_PATH=pyzoo

echo ">>> $MASTER, start spark-driver"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="2-5" \
      --oom-kill-disable \
      --device=/dev/gsgx \
      --device=/dev/sgx/enclave \
      --device=/dev/sgx/provision \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
      -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
      -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
      -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
      -v $PYZOO_PATH:/ppml/trusted-big-data-ml/work/pyzoo \
      -v $ZOO_PATH:/ppml/trusted-big-data-ml/work/zoo \
      --name=spark-driver \
      -e LOCAL_IP=$MASTER \
      -e SGX_MEM_SIZE=32G \
      -e SPARK_MASTER=spark://$MASTER:7077 \
      -e SPARK_DRIVER_PORT=10027 \
      -e SPARK_DRIVER_BLOCK_MANAGER_PORT=10026 \
      $TRUSTED_BIGDATA_ML_DOCKER bash"
