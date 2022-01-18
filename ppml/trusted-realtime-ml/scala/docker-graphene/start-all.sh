#!/bin/bash
set -x

cd /ppml/trusted-realtime-ml/redis
export SGX_MEM_SIZE=16G
./init.sh
echo "redis initiated"

cd /ppml/trusted-realtime-ml/java
export SGX_MEM_SIZE=32G
./init.sh
echo "java initiated"

export REDIS_HOST=$LOCAL_IP
./init-cluster-serving.sh
echo "cluster serving initiated"

cd /ppml/trusted-realtime-ml/redis
./start-redis.sh &
echo "redis started"
{ set +x; } 2>/dev/null
bash /ppml/trusted-realtime-ml/check-status.sh redis
set -x

cd /ppml/trusted-realtime-ml/java
export FLINK_JOB_MANAGER_IP=$LOCAL_IP
./start-flink-jobmanager.sh &
echo "flink-jobmanager started"
{ set +x; } 2>/dev/null
bash /ppml/trusted-realtime-ml/check-status.sh flinkjm
set -x

export FLINK_TASK_MANAGER_IP=$LOCAL_IP
while ! nc -z -v $FLINK_TASK_MANAGER_IP $FLINK_JOB_MANAGER_REST_PORT 2>&1 | egrep -a "open" | wc -l; do
  sleep 1
done
./start-flink-taskmanager.sh &
echo "flink-taskmanager started"
{ set +x; } 2>/dev/null
bash /ppml/trusted-realtime-ml/check-status.sh flinktm
set -x

while ! nc -z -v $REDIS_HOST $REDIS_PORT 2>&1 | egrep -a "open" | wc -l; do
  sleep 1
done
./start-http-frontend.sh &
echo "http-frontend started"
{ set +x; } 2>/dev/null
bash /ppml/trusted-realtime-ml/check-status.sh frontend
set -x

while ! nc -z -v $FLINK_TASK_MANAGER_IP $FLINK_TASK_MANAGER_DATA_PORT 2>&1 | egrep -a "open" | wc -l; do
  sleep 1
done
./start-cluster-serving-job.sh &
echo "cluster-serving-job started"
{ set +x; } 2>/dev/null
bash /ppml/trusted-realtime-ml/check-status.sh cluster

bash /ppml/trusted-realtime-ml/check-status.sh
