#!/bin/bash

#set -x

source ./environment.sh

echo ">>> $MASTER"
ssh root@$MASTER "docker rm -f spark-master"

#ssh root@$MASTER "docker rm -f spark-driver"

bash ./stop-distributed-standalone-spark.sh

for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker rm -f spark-worker-$worker"
  done
