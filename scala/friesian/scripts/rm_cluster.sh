#!/bin/bash
set -e

if (( $# < 1)); then
  echo "Usage: service_type"
  echo "Usage example: ranking"
  echo "service_type can be one of ranking, recall, recommender, feature and feature_recall."
  exit -1
fi

source config.sh

service=$1

if [[ -z "$CLUSTER_NAME" ]]; then
    CLUSTER_NAME="friesian"
fi

if [ "$service" == "ranking" ]; then
    SERVER_LIST=("${RANKING_SERVER_LIST[@]}")
elif [ "$service" == "feature" ]; then
    SERVER_LIST=("${FEATURE_SERVER_LIST[@]}")
elif [ "$service" == "feature_recall" ]; then
    SERVER_LIST=("${FEATURE_RECALL_SERVER_LIST[@]}")
elif [ "$service" == "recall" ]; then
    SERVER_LIST=("${RECALL_SERVER_LIST[@]}")
elif [ "$service" == "recommender" ]; then
    SERVER_LIST=("${RECOMMENDER_SERVER_LIST[@]}")
else
    echo "Unsupported service_type, service_type can be one of ranking, recall, recommender, feature and feature_recall."
fi

cmd="docker ps --filter name=$CLUSTER_NAME-$service* --filter status=running -aq | xargs docker stop; docker ps -a -q --filter name=$CLUSTER_NAME-$service* |  xargs docker rm"
# echo $cmd

for i in "${SERVER_LIST[@]}"; do
    echo Stopping $service service on $i
    if [[ -z "$SERVER_PASS" ]]; then
        ssh $i "$cmd || true"
    else
        sshpass -p $SERVER_PASS ssh $i "$cmd || true"
    fi
done

