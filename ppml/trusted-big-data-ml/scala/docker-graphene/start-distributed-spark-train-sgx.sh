#!/bin/bash
set -x

source ./environment.sh

echo ">>> $MASTER, start spark-driver"
ssh root@$MASTER "docker run -d \
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
      --name=spark-driver \
      -e LOCAL_IP=$MASTER \
      -e SGX_MEM_SIZE=32G \
      -e SPARK_MASTER=spark://$MASTER:7077 \
      -e SPARK_DRIVER_PORT=10027 \
      -e SPARK_DRIVER_BLOCK_MANAGER_PORT=10026 \
      $TRUSTED_BIGDATA_ML_DOCKER bash -c 'cd /ppml/trusted-big-data-ml && ./init.sh && ./start-spark-standalone-driver-sgx.sh'"
while ! ssh root@$MASTER "docker logs spark-driver | grep 'model saved'"; do
  sleep 100
done
echo ">>> $MASTER, cluster-serving started successfully."

