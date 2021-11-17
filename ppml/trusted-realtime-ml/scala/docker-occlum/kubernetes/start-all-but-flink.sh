#!/bin/bash
set -x

cd /opt/redis
export SGX_MEM_SIZE=16G
test "$SGX_MODE" = sgx && ./init.sh
echo "redis initiated"


cd /ppml/trusted-realtime-ml/java
export SGX_MEM_SIZE=32G
test "$SGX_MODE" = sgx && ./init.sh
echo "java initiated"


export REDIS_HOST=redis-service
./init-cluster-serving.sh
echo "cluster serving initiated"


cd /opt
./start-redis.sh &


while ! nc -z $REDIS_HOST $REDIS_PORT; do
  sleep 5
done
echo "redis started"


./start-http-frontend.sh &
echo "http-frontend started"


while ! nc -z $LOCAL_IP 10020; do
  sleep 1
done
./start-cluster-serving-job.sh
echo "cluster-serving-job started"


bash /opt/check-status.sh redis frontend serving
