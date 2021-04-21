#!/bin/bash

set -x

REDIS_MAX_LOOP_TIME=60
HTTP_FRONTEND_MAX_LOOP_TIME=270
CLUSTER_SERVING_MAX_LOOP_TIME=150

source environment.sh

echo "### phase.1 distribute the keys and password"
echo ">>> $MASTER"
ssh root@$MASTER "rm -rf $ENCLAVE_KEY_PATH && rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && mkdir -p $AZ_PPML_PATH"
scp -r $SOURCE_ENCLAVE_KEY_PATH root@$MASTER:$ENCLAVE_KEY_PATH
scp -r $SOURCE_KEYS_PATH root@$MASTER:$KEYS_PATH
scp -r $SOURCE_SECURE_PASSWORD_PATH root@$MASTER:$SECURE_PASSWORD_PATH
for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "rm -rf $ENCLAVE_KEY_PATH && rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && mkdir -p $AZ_PPML_PATH"
    scp -r $SOURCE_ENCLAVE_KEY_PATH root@$worker:$ENCLAVE_KEY_PATH
    scp -r $SOURCE_KEYS_PATH root@$worker:$KEYS_PATH
    scp -r $SOURCE_SECURE_PASSWORD_PATH root@$worker:$SECURE_PASSWORD_PATH
  done
echo "### phase.1 distribute the keys and password finished successfully"

echo "### phase.2 pull the docker image"
echo ">>> $MASTER"
ssh root@$MASTER "docker pull $TRUSTED_CLUSTER_SERVING_DOCKER"
for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker pull $TRUSTED_CLUSTER_SERVING_DOCKER"
  done
echo "### phase.2 pull the docker image finished successfully"


echo "### phase.3 deploy the cluster serving components"
echo ">>> $MASTER, start redis"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="0-2" \
      --oom-kill-disable \
      --device=/dev/gsgx \
      --device=/dev/sgx/enclave \
      --device=/dev/sgx/provision \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
      -v $KEYS_PATH:/ppml/trusted-realtime-ml/redis/work/keys \
      -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/redis/work/password \
      --name=redis \
      -e SGX_MEM_SIZE=16G \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /ppml/trusted-realtime-ml/redis && ./init-redis.sh && ./start-redis.sh'"

REDIS_ELAPSED_TIME=0
while ! ssh root@$MASTER "nc -z $MASTER 6379"; do
    { set +x; } 2>/dev/null
    if [ $REDIS_ELAPSED_TIME -gt $REDIS_MAX_LOOP_TIME ] ; then 
        echo "Error: Redis port 6379 is unavailable."
        break
    fi
    REDIS_ELAPSED_TIME=$((REDIS_ELAPSED_TIME+10))
    set -x
    sleep 10
done
{ set +x; } 2>/dev/null
if [ $REDIS_ELAPSED_TIME -le $REDIS_MAX_LOOP_TIME ] ; then 
    echo ">>> $MASTER, redis started successfully."
fi
set -x


# Flink jobmanager and taskmanager are in deploy-flink.sh.

bash deploy-distributed-flink.sh 

echo ">>> $MASTER, start http-frontend"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="31-32" \
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
      --name=http-frontend \
      -e SGX_MEM_SIZE=32G \
      -e REDIS_HOST=$MASTER \
      -e CORE_NUM=2 \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /ppml/trusted-realtime-ml/java && ./init-java.sh && ./start-http-frontend.sh'"

HTTP_FRONTEND_ELAPSED_TIME=0
while ! ssh root@$MASTER "nc -z $MASTER 10023"; do
    { set +x; } 2>/dev/null
    if [ $HTTP_FRONTEND_ELAPSED_TIME -gt $HTTP_FRONTEND_MAX_LOOP_TIME ] ; then 
        echo "Error: http frontend port 10023 is unavailable."
        break
    fi
    HTTP_FRONTEND_ELAPSED_TIME=$((HTTP_FRONTEND_ELAPSED_TIME+10))
    set -x
    sleep 10
done
{ set +x; } 2>/dev/null
if [ $HTTP_FRONTEND_ELAPSED_TIME -le $HTTP_FRONTEND_MAX_LOOP_TIME ] ; then 
    echo ">>> $MASTER, http-frontend started successfully."
fi
set -x

echo ">>> $MASTER, start cluster-serving"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="33-34" \
      --oom-kill-disable \
      --device=/dev/gsgx \
      --device=/dev/sgx/enclave \
      --device=/dev/sgx/provision \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
      -v $KEYS_PATH:/ppml/trusted-realtime-ml/java/work/keys \
      -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/redis/work/password \
      --name=cluster-serving \
      -e SGX_MEM_SIZE=16G \
      -e REDIS_HOST=$MASTER \
      -e CORE_NUM=2 \
      -e FLINK_JOB_MANAGER_IP=$MASTER \
      -e FLINK_JOB_MANAGER_REST_PORT=8081 \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /ppml/trusted-realtime-ml/java && ./init-cluster-serving.sh && ./start-cluster-serving-job.sh'"

CLUSTER_SERVING_ELAPSED_TIME=0
while ! ssh root@$MASTER "docker logs cluster-serving | grep 'Job has been submitted'"; do
    { set +x; } 2>/dev/null
    if [ $CLUSTER_SERVING_ELAPSED_TIME -gt $CLUSTER_SERVING_MAX_LOOP_TIME ] ; then 
        echo "Error: cluster-serving timeout."
        break
    fi
    CLUSTER_SERVING_ELAPSED_TIME=$((CLUSTER_SERVING_ELAPSED_TIME+10))
    set -x
    sleep 10
done
{ set +x; } 2>/dev/null
if [ $CLUSTER_SERVING_ELAPSED_TIME -le $CLUSTER_SERVING_MAX_LOOP_TIME ] ; then 
    echo ">>> $MASTER, cluster-serving started successfully."
fi
