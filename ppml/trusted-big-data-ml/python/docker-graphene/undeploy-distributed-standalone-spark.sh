#!/bin/bash

#set -x

source ./environment.sh

echo ">>> $MASTER"
ssh root@$MASTER "docker rm -f spark-master"

for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker rm -f spark-worker-$worker"
  done

echo "Undeploying completed."