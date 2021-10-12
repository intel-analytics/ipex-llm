#!/bin/bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
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

execute_ray_test tcmf_elctricity "${BIGDL_ROOT}/python/chronos/example/tcmf/run_electricity.py --use_dummy_data --smoke"
time1=$?

echo "#1 chronos tcmf example time used:$time1 seconds"

clear_up