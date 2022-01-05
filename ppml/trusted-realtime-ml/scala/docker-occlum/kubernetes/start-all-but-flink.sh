#!/bin/bash
set -x


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
