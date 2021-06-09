#!/bin/bash
# Acceptable arguments: master, worker, all

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


if [ "$#" -gt 2 ]; then
    echo "Acceptable arguments: \"all\", or one or more among \"master\", \"worker\"."
elif [ "$all" -eq 1 ]; then
    ssh root@$MASTER "docker exec spark-master bash /ppml/trusted-big-data-ml/check-status.sh master"
    for worker in ${WORKERS[@]}
    do
        ssh root@$worker "docker exec spark-worker-$worker bash /ppml/trusted-big-data-ml/check-status.sh worker"
    done
else
    for arg in "$@"
    do
        if [ "$arg" == master ]; then
            ssh root@$MASTER "docker exec spark-master bash /ppml/trusted-big-data-ml/check-status.sh master"
        elif [ "$arg" == worker ]; then
            for worker in ${WORKERS[@]}
            do
                ssh root@$worker "docker exec spark-worker-$worker bash /ppml/trusted-big-data-ml/check-status.sh worker"
            done
        else
            echo "Acceptable arguments: \"all\", or one or more among \"master\", \"worker\"."
        fi
    done
fi
