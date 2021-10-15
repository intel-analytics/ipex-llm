#!/bin/bash

#set -x

source ./environment.sh

echo "### phase.1 distribute the keys and password and data"
echo ">>> $MASTER"
ssh root@$MASTER "rm -rf $ENCLAVE_KEY_PATH && rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && rm -rf $DATA_PATH && mkdir -p $AZ_PPML_PATH"
scp -r $SOURCE_ENCLAVE_KEY_PATH root@$MASTER:$ENCLAVE_KEY_PATH
scp -r $SOURCE_KEYS_PATH root@$MASTER:$KEYS_PATH
scp -r $SOURCE_SECURE_PASSWORD_PATH root@$MASTER:$SECURE_PASSWORD_PATH
scp -r $SOURCE_DATA_PATH root@$MASTER:$DATA_PATH
for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "rm -rf $ENCLAVE_KEY_PATH && rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && rm -rf $DATA_PATH && mkdir -p $AZ_PPML_PATH"
    scp -r $SOURCE_ENCLAVE_KEY_PATH root@$worker:$ENCLAVE_KEY_PATH
    scp -r $SOURCE_KEYS_PATH root@$worker:$KEYS_PATH
    scp -r $SOURCE_SECURE_PASSWORD_PATH root@$worker:$SECURE_PASSWORD_PATH
    scp -r $SOURCE_DATA_PATH root@$worker:$DATA_PATH
  done
echo "### phase.1 distribute the keys and password finished successfully"

echo "### phase.2 pull the docker image"
echo ">>> $MASTER"
ssh root@$MASTER "docker pull $TRUSTED_BIGDATA_ML_DOCKER"
for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker pull $TRUSTED_BIGDATA_ML_DOCKER"
  done
echo "### phase.2 pull the docker image finished successfully"

echo "### phase.3 deploy the spark components"
echo ">>> $MASTER, start spark master"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="0-1" \
      --oom-kill-disable \
      --device=/dev/gsgx \
      --device=/dev/sgx/enclave \
      --device=/dev/sgx/provision \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
      -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
      -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
      --name=spark-master \
      -e LOCAL_IP=$MASTER \
      -e SGX_MEM_SIZE=16G \
      -e SPARK_MASTER_IP=$MASTER \
      -e SPARK_MASTER_PORT=7077 \
      -e SPARK_MASTER_WEBUI_PORT=8080 \
      $TRUSTED_BIGDATA_ML_DOCKER bash -c 'cd /ppml/trusted-big-data-ml && ./init.sh && ./start-spark-standalone-master-sgx.sh  && tail -f /dev/null'"
while ! ssh root@$MASTER "nc -z $MASTER 8080"; do
  sleep 10
done

for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker run -itd \
          --privileged \
          --net=host \
          --cpuset-cpus="6-10" \
          --oom-kill-disable \
          --device=/dev/gsgx \
          --device=/dev/sgx/enclave \
          --device=/dev/sgx/provision \
          -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
          -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
          -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
          -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
          --name=spark-worker-$worker \
          -e LOCAL_IP=$worker \
          -e SGX_MEM_SIZE=64G \
          -e SPARK_MASTER=spark://$MASTER:7077 \
          -e SPARK_WORKER_PORT=8082 \
          -e SPARK_WORKER_WEBUI_PORT=8081 \
          $TRUSTED_BIGDATA_ML_DOCKER bash -c 'cd /ppml/trusted-big-data-ml && ./init.sh && ./start-spark-standalone-worker-sgx.sh'"
  done

for worker in ${WORKERS[@]}
  do
    while ! ssh root@$worker "nc -z $worker 8081"; do
      sleep 10
    done
  done

./distributed-check-status.sh
