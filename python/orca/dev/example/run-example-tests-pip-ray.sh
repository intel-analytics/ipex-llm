#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling BigDL"
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

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

execute_ray_test auto-estimator-pytorch "${BIGDL_ROOT}/python/orca/example/automl/autoestimator/autoestimator_pytorch.py --trials 5 --epochs 2"
time1=$?

if [ -f ${BIGDL_ROOT}/data/airline_14col.data ]
then
    echo "airline_14col.data already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${BIGDL_ROOT}/data/
fi

execute_ray_test auto-xgboost-classifier "${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostClassifier.py -p ${BIGDL_ROOT}/data/airline_14col.data"
time2=$?

if [ -f ${BIGDL_ROOT}/data/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${BIGDL_ROOT}/data/
fi

execute_ray_test auto-xgboost-regressor "${BIGDL_ROOT}/python/orca/example/automl/autoxgboost/AutoXGBoostRegressor.py -p ${BIGDL_ROOT}/data/incd.csv"
time3=$?

echo "#1 auto-estimator-pytorch time used:$time1 seconds"
echo "#2 auto-xgboost-classifier time used:$time2 seconds"
echo "#3 auto-xgboost-regressor time used:$time3 seconds"

clear_up
