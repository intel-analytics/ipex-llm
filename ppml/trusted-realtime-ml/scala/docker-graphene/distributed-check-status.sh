#!/bin/bash
# Acceptable arguments: redis, flinkjm, flinktm, frontend, serving, all

source environment.sh

all=0
if [ "$#" -lt 1 ]; then
    echo "No argument passed, detecting all component states."
    all=$((all+1))
else
    for arg in "$@"
    do
        if [ "$arg" == all ]; then
            echo "Detecting all component states."
            all=$((all+1))
            break
        fi
    done
fi


if [ "$#" -gt 5 ]; then
    echo "Acceptable arguments: \"all\", or one or more among \"redis\", \"flinkjm\", \"flinktm\", \"frontend\", \"serving\"."
elif [ "$all" -eq 1 ]; then 
    ssh root@$MASTER "docker exec redis bash /ppml/trusted-realtime-ml/check-status.sh redis"
    ssh root@$MASTER "docker exec flink-job-manager bash /ppml/trusted-realtime-ml/check-status.sh flinkjm"  
    for worker in ${WORKERS[@]}
    do
        ssh root@$worker "docker exec flink-task-manager-$worker bash /ppml/trusted-realtime-ml/check-status.sh flinktm"
    done
    ssh root@$MASTER "docker exec http-frontend bash /ppml/trusted-realtime-ml/check-status.sh frontend"
    ssh root@$MASTER "docker exec cluster-serving bash /ppml/trusted-realtime-ml/check-status.sh serving"
else 
    for arg in "$@"
    do
        if [ "$arg" == redis ]; then
            ssh root@$MASTER "docker exec redis bash /ppml/trusted-realtime-ml/check-status.sh redis"
        elif [ "$arg" == flinkjm ]; then
            ssh root@$MASTER "docker exec flink-job-manager bash /ppml/trusted-realtime-ml/check-status.sh flinkjm"  
        elif [ "$arg" == flinktm ]; then
            for worker in ${WORKERS[@]}
            do
                ssh root@$worker "docker exec flink-task-manager-$worker bash /ppml/trusted-realtime-ml/check-status.sh flinktm"
            done
        elif [ "$arg" == frontend ]; then
            ssh root@$MASTER "docker exec http-frontend bash /ppml/trusted-realtime-ml/check-status.sh frontend"
        elif [ "$arg" == serving ]; then
            ssh root@$MASTER "docker exec cluster-serving bash /ppml/trusted-realtime-ml/check-status.sh serving"
        else 
            echo "Acceptable arguments: \"all\", or one or more among \"redis\", \"flinkjm\", \"flinktm\", \"frontend\", \"serving\"."
        fi
    done
fi
