#!/bin/bash
set -x

cd /ppml/trusted-cluster-serving/redis
export SGX_MEM_SIZE=16G
./init-redis.sh
echo "redis inited"

cd /ppml/trusted-cluster-serving/java
export SGX_MEM_SIZE=32G
./init-java.sh
echo "java inited"

export REDIS_HOST=$LOCAL_IP
./init-cluster-serving.sh
echo "cluster serving inited"

cd /ppml/trusted-cluster-serving/redis
./start-redis.sh &
echo "redis started"

cd /ppml/trusted-cluster-serving/java
export FLINK_JOB_MANAGER_IP=$LOCAL_IP
./start-flink-jobmanager.sh &
echo "flink-jobmanager started"

export FLINK_TASK_MANAGER_IP=$LOCAL_IP
while ! nc -z $FLINK_TASK_MANAGER_IP $FLINK_JOB_MANAGER_REST_PORT; do
  sleep 1
done
./start-flink-taskmanager.sh &
echo "flink-taskmanager started"

while ! nc -z $REDIS_HOST $REDIS_PORT; do
  sleep 1
done
./start-http-frontend.sh &
echo "http-frontend started"

while ! nc -z $FLINK_TASK_MANAGER_IP $FLINK_TASK_MANAGER_DATA_PORT; do
  sleep 1
done
./start-cluster-serving-job.sh &
echo "cluster-serving-job started"
