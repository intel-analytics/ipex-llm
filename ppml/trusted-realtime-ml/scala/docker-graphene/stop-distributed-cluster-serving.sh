#!/bin/bash

set -x

source ./environment.sh

echo ">>> $MASTER"
ssh root@$MASTER "docker rm -f redis"
ssh root@$MASTER "docker rm -f http-frontend"
ssh root@$MASTER "docker rm -f cluster-serving"


# Use the following script to stop flink jobmanager and taskmanager.
bash undeploy-distributed-flink.sh
