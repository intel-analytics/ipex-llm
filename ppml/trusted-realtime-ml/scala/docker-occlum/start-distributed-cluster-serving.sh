#!/bin/bash

set -x

source ./environment.sh

echo "### phase.1 distribute the keys and password"
echo ">>> $MASTER"
ssh root@$MASTER "rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && mkdir -p $AZ_PPML_PATH"
scp -r $SOURCE_KEYS_PATH root@$MASTER:$KEYS_PATH
scp -r $SOURCE_SECURE_PASSWORD_PATH root@$MASTER:$SECURE_PASSWORD_PATH
for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && mkdir -p $AZ_PPML_PATH"
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
      --name=redis \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /opt && ./start-redis.sh'"
while ! ssh root@$MASTER "nc -z $MASTER 6379"; do
  sleep 10
done
echo ">>> $MASTER, redis started successfully."

# Script for starting flink job manager and task manager is in the following file:
bash ./deploy-flink.sh

echo ">>> $MASTER, start http-frontend"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="31-32" \
      --oom-kill-disable \
      --device=/dev/sgx \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $KEYS_PATH:/opt/keys \
      -v $SECURE_PASSWORD_PATH:/opt/password \
      --name=http-frontend \
      -e SGX_MEM_SIZE=32G \
      -e REDIS_HOST=$MASTER \
      -e CORE_NUM=2 \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /opt && ./start-http-frontend.sh'"
while ! ssh root@$MASTER "nc -z $MASTER 10023"; do
  sleep 10
done
echo ">>> $MASTER, http-frontend started successfully."

echo ">>> $MASTER, start cluster-serving"
ssh root@$MASTER "docker run -itd \
      --privileged \
      --net=host \
      --cpuset-cpus="33-34" \
      --oom-kill-disable \
      -v $KEYS_PATH:/opt/keys \
      -v $SECURE_PASSWORD_PATH:/opt/password \
      --name=cluster-serving \
      -e REDIS_HOST=$MASTER \
      -e CORE_NUM=2 \
      -e FLINK_JOB_MANAGER_IP=$MASTER \
      -e FLINK_JOB_MANAGER_REST_PORT=8081 \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /opt && ./start-cluster-serving-job.sh'"
while ! ssh root@$MASTER "docker logs cluster-serving | grep 'Job has been submitted'"; do
  sleep 10
done
echo ">>> $MASTER, cluster-serving started successfully."
