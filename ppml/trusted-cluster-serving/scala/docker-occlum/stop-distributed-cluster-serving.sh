#!/bin/bash

set -x

source ./environment.sh

echo ">>> $MASTER"
ssh root@$MASTER "docker rm -f redis"
ssh root@$MASTER "docker rm -f flink-job-manager"
ssh root@$MASTER "docker rm -f http-frontend"
ssh root@$MASTER "docker rm -f cluster-serving"

for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker rm -f flink-task-manager-$worker"
  done
