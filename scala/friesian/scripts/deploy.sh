#!/bin/bash
set -e

if (( $# < 3)); then
  echo "Usage: service_type target_port published_port config_name"
  echo "Usage example: ranking 8083 7083 config_ranking.yaml"
  echo "service_type can be one of ranking, recall, recommender, feature and feature_recall."
  exit -1
fi

source config.sh

echo "Docker image tag: $IMAGE_TAG"
# echo $VOLUME_CONF

service=$1
target_port=$2
published_port=$3
config_name=$4
other_param=""
instance_per_node=1
cpuset=()

if [ $RANKING_INSTANCE_PER_NODE -lt 1 ]; then
    echo "RANKING_INSTANCE_PER_NODE >= 1, but got $RANKING_INSTANCE_PER_NODE"
    exit 1
fi

if [ $RECOMMENDER_INSTANCE_PER_NODE -lt 1 ]; then
    echo "RECOMMENDER_INSTANCE_PER_NODE >= 1, but got $RECOMMENDER_INSTANCE_PER_NODE"
    exit 1
fi

if [ $RECALL_INSTANCE_PER_NODE -lt 1 ]; then
    echo "RECALL_INSTANCE_PER_NODE >= 1, but got $RECALL_INSTANCE_PER_NODE"
    exit 1
fi

if [[ -z "$config_name" ]]; then
    config_name="config_$service.yaml"
fi

if [[ -z "$CLUSTER_NAME" ]]; then
    CLUSTER_NAME="friesian"
fi


if [ "$service" == "ranking" ]; then
    SERVER_LIST=("${RANKING_SERVER_LIST[@]}")
    if [[ -n "$OMP_NUM_THREADS" ]]; then
        other_param="$other_param -e OMP_NUM_THREADS=$OMP_NUM_THREADS"
    fi
    instance_per_node=$RANKING_INSTANCE_PER_NODE
    params="ranking -c $VOLUME_TARGET/$config_name"
elif [ "$service" == "feature" ]; then
    SERVER_LIST=("${FEATURE_SERVER_LIST[@]}")
    params="feature -c $VOLUME_TARGET/$config_name"
elif [ "$service" == "feature_recall" ]; then
    SERVER_LIST=("${FEATURE_RECALL_SERVER_LIST[@]}")
    params="feature -c $VOLUME_TARGET/$config_name"
elif [ "$service" == "recall" ]; then
    SERVER_LIST=("${RECALL_SERVER_LIST[@]}")
    other_param="$other_param -e OMP_NUM_THREADS=48"
    instance_per_node=$RECALL_INSTANCE_PER_NODE
    params="recall -c $VOLUME_TARGET/$config_name"
elif [ "$service" == "recommender" ]; then
    SERVER_LIST=("${RECOMMENDER_SERVER_LIST[@]}")
    instance_per_node=$RECOMMENDER_INSTANCE_PER_NODE
    params="recommender -c $VOLUME_TARGET/$config_name"
else
    echo "Unsupported service_type, service_type can be one of ranking, recall, recommender, feature and client."
fi

function start_multi_contianers {
    _ssh_cmd=$1
    _core_num=$2

    core_per_instance=$((core_num / instance_per_node))
    p_port=$published_port
    t_port=$target_port
    for ((i = 0; i < $((instance_per_node)); i ++)); do
        core_start=$(( i * core_per_instance ))
        core_end=$(( (i + 1) * core_per_instance - 1 ))
        cmd2="docker run -it -d -p $p_port:$t_port --cpuset-cpus $core_start-$core_end --name $CLUSTER_NAME-$service-$RANDOM $other_param --mount $VOLUME_CONF $IMAGE_TAG $params -p $t_port"
        echo ${cmd2} 
        $_ssh_cmd $cmd2 || true
        p_port=$(( p_port + 1 ))
        t_port=$(( t_port + 1 ))
    done
}


for i in "${SERVER_LIST[@]}"; do
    echo Starting $service service on $i
    if [ $instance_per_node -gt 1 ]; then
        if [[ -z "$SERVER_PASS" ]]; then
            core_num=`ssh $i "nproc"`
            echo "core_num:$core_num"
            start_multi_contianers "ssh $i" "$core_num"
        else
            core_num=`sshpass -p $SERVER_PASS ssh $i "nproc"`
            echo "core_num:$core_num"
            start_multi_contianers "sshpass -p $SERVER_PASS ssh $i" "$core_num"
        fi
    else
        cmd="docker run -it -d -p $published_port:$target_port --name $CLUSTER_NAME-$service-$RANDOM $other_param --mount $VOLUME_CONF $IMAGE_TAG $params -p $target_port"
        echo $cmd
        if [[ -z "$SERVER_PASS" ]]; then
            ssh $i "$cmd" || true
        else
            sshpass -p $SERVER_PASS ssh $i "$cmd" || true
        fi
    fi
done

