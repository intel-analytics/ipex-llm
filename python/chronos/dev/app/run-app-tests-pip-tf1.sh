#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling bigdl"
    pip uninstall -y bigdl-chronos
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

chmod +x ${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh

set -e

echo "#1 start app test for chronos-network-traffic-autots-forecasting-deprecated"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated

FILENAME="${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated.py
sed -i 's/epochs=2/epochs=1/g; s/object_store_memory=\"1/object_store_memory=\"10/' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/

python ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-autots-forecasting-deprecated failed"
    exit $exit_status
fi
now=$(date "+%s")
time1=$((now-start))
echo "#1 chronos-network-traffic-autots-forecasting-deprecated time used:$time1 seconds"

echo "#2 start app test for chronos-network-traffic-model-forecasting"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting

FILENAME="${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
sed -i "s/epochs=50/epochs=2/g; s/epochs=20/epochs=2/g" ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/

python ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-model-forecasting failed"
    exit $exit_status
fi
now=$(date "+%s")
time2=$((now-start))
echo "#2 chronos-network-traffic-model-forecasting time used:$time2 seconds"

echo "#4 start app test for chronos-anomaly-detect-unsupervised-forecast-based"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based

wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -O ${BIGDL_ROOT}/python/chronos/use-case/AIOps/m_1932.csv
echo "Finished downloading AIOps data"
#FILENAME="${BIGDL_ROOT}/python/chronos/use-case/AIOps/m_1932.csv"
#if [ -f "$FILENAME" ]
#then
#   echo "$FILENAME already exists."
#else
#   echo "Downloading AIOps data"
#
#   wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -P ${BIGDL_ROOT}/python/chronos/use-case/AIOps
#
#   echo "Finished downloading AIOps data"
#fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
sed -i "s/epochs=20/epochs=2/g" ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
cd ${BIGDL_ROOT}/python/chronos/use-case/AIOps/

python ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-anomaly-detect-unsupervised-forecast-based failed"
    exit $exit_status
fi
now=$(date "+%s")
time4=$((now-start))
echo "#4 chronos-anomaly-detect-unsupervised-forecast-based time used:$time4 seconds"

# This should be done at the very end after all tests finish.
clear_up
