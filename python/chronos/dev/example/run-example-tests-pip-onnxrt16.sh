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

if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
  mv ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv
fi

execute_ray_test onnx_autotsestimator_nyc_taxi "${BIGDL_ROOT}/python/chronos/example/onnx/onnx_autotsestimator_nyc_taxi.py"
time5=$?

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
  mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

execute_ray_test onnx_forecaster_network_traffic "${BIGDL_ROOT}/python/chronos/example/onnx/onnx_forecaster_network_traffic.py"
time6=$?

if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
  mv ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv
fi

sed -i 's/epochs=10/epochs=1/' "${BIGDL_ROOT}/python/chronos/example/quantization/quantization_tcnforecaster_nyc_taxi.py"
execute_ray_test quantization_tcnforecaster_nyc_taxi "${BIGDL_ROOT}/python/chronos/example/quantization/quantization_tcnforecaster_nyc_taxi.py"
time7=$?

echo "#5 onnx_autotsestimator_nyc_taxi time used:$time5 seconds"
echo "#6 onnx_forecaster_network_traffic used:$time6 seconds"
echo "#7 quantization_tcnforecaster_nyc_taxi used:$time7 seconds"
