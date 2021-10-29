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


if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
fi

execute_ray_test autolstm_nyc_taxi "${BIGDL_ROOT}/python/chronos/example/auto_model/autolstm_nyc_taxi.py"
time1=$?

if [ ! -f {BIGDL_ROOT}/pyzoo/zoo/chronos/examples/auto_model/nyc_taxi.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ${BIGDL_ROOT}/python/chronos/example/auto_model/
fi

execute_ray_test autoprophet_nyc_taxi "${BIGDL_ROOT}/python/chronos/example/auto_model/autoprophet_nyc_taxi.py --datadir ${BIGDL_ROOT}/python/chronos/example/auto_model/nyc_taxi.csv"
time2=$?

if [ ! -f {BIGDL_ROOT}/pyzoo/zoo/chronos/examples/simulator/data_train_small.npz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/doppelGANger_data/data_train_small.npz -P ${BIGDL_ROOT}/python/chronos/example/simulator/
fi

execute_ray_test dpgansimulator_wwt "${BIGDL_ROOT}/python/chronos/example/simulator/dpgansimulator_wwt.py --datadir ${BIGDL_ROOT}/python/chronos/example/simulator/data_train_small.npz --epoch 1 --plot_figures False"
time3=$?

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
  mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

execute_ray_test distributed_training_network_traffic "${BIGDL_ROOT}/python/chronos/example/distributed/distributed_training_network_traffic.py"
time4=$?

if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
  mv ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv
fi

# When the thread of onnxruntime is None, "pthread_setaffinity_np failed" may appear.
sed -i '/onnx/d' ${BIGDL_ROOT}/python/chronos/example/onnx/onnx_autotsestimator_nyc_taxi.py

execute_ray_test onnx_autotsestimator_nyc_taxi "${BIGDL_ROOT}/python/chronos/example/onnx/onnx_autotsestimator_nyc_taxi.py"
time5=$?

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
  mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

# When the thread of onnxruntime is None, "pthread_setaffinity_np failed" may appear.
sed -i '/onnx/d' ${BIGDL_ROOT}/python/chronos/example/onnx/onnx_forecaster_network_traffic.py

execute_ray_test onnx_forecaster_network_traffic "${BIGDL_ROOT}/python/chronos/example/onnx/onnx_forecaster_network_traffic.py"
time6=$?

echo "#1 autolstm_nyc_taxi time used:$time1 seconds"
echo "#2 autoprophet_nyc_taxi time used:$time2 seconds"
echo "#3 dpgansimulator_wwt time used:$time3 seconds"
echo "#4 distributed_training_network_traffic time used:$time4 seconds"
echo "#5 onnx_autotsestimator_nyc_taxi time used:$time5 seconds"
echo "#6 onnx_forecaster_network_traffic used:$time6 seconds"

clear_up
