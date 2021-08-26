#!/bin/bash
set -x
export FLINK_JOB_MANAGER_IP=flink-jobmanager
cd /ppml/trusted-realtime-ml/redis
export SGX_MEM_SIZE=16G
test "$SGX_MODE" = sgx && ./init-redis.sh
echo "redis initiated"


cd /ppml/trusted-realtime-ml/java
export SGX_MEM_SIZE=32G
test "$SGX_MODE" = sgx && ./init-java.sh
echo "java initiated"


export REDIS_HOST=$LOCAL_IP
./init-cluster-serving.sh
echo "cluster serving initiated"


cd /ppml/trusted-realtime-ml/redis
./start-redis.sh &


while ! nc -z $REDIS_HOST $REDIS_PORT; do
  sleep 5
done
echo "redis started"


cd ../java
./start-http-frontend.sh &
echo "http-frontend started"


while ! nc -z $LOCAL_IP 10020; do
  sleep 1
done
./start-cluster-serving-job.sh &
echo "cluster-serving-job started"


bash /ppml/trusted-realtime-ml/check-status.sh
