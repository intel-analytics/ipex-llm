#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling bigdl"
    pip uninstall -y bigdl-chronos
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}
#if image exist this two dependency, remove below
execute_ray_test(){
    echo "start example $1"
    start=$(date "+%s")
    python $2
    exit_status=$?
    if [ $exit_status -ne 0 ];
    then
        clear_up
        echo "$1 failed"
        exit $exit_status
    fi
    now=$(date "+%s")
    return $((now-start))
}

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
  mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

execute_ray_test distributed_training_network_traffic "${BIGDL_ROOT}/python/chronos/example/distributed/distributed_training_network_traffic.py"
time1=$?

ray stop -f
ray start --head

execute_ray_test distributed_training_network_traffic "${BIGDL_ROOT}/python/chronos/example/distributed/distributed_training_network_traffic.py --runtime ray --address localhost:6379"
time2=$?

ray stop -f