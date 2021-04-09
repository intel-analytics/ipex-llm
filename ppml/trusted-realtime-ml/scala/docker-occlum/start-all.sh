#!/bin/bash
set -x

bash /opt/occlum/start_aesm.sh
echo "Starting AESM service..."

cd /opt
./init-occlum-taskmanager.sh
echo "occlum flink jobmanager image built"

cd /opt
./start-redis.sh &
echo "redis started"

export FLINK_JOB_MANAGER_IP=$LOCAL_IP
./start-flink-jobmanager.sh &
echo "flink-jobmanager started"


[ "$(pgrep aesm)" ] && echo "AESM started" || echo "AESM not started"
while ! pgrep aesm; do
  sleep 10
done
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

./check-status.sh
