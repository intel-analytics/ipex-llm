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

echo "#5 start app test for chronos-anomaly-detect-unsupervised"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised

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

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.py
cd ${BIGDL_ROOT}/python/chronos/use-case/AIOps/

python ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-anomaly-detect-unsupervised failed"
    exit $exit_status
fi
now=$(date "+%s")
time5=$((now-start))
echo "#5 chronos-anomaly-detect-unsupervised time used:$time5 seconds"

echo "#6 start app test for chronos-stock-prediction"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction.py
sed -i "s/epochs\ =\ 50/epochs\ =\ 2/g; s/batch_size\ =\ 16/batch_size\ =\ 108/g" ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction.py
cwd=$PWD
cd ${BIGDL_ROOT}/python/chronos/use-case/fsi/

# download data
if [ -d "data" ]
then
    echo "data already exists"
else
    echo "Downloading stock prediction data"

    mkdir data
    cd data
    wget https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip
    wget https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/merge.sh
    chmod +x merge.sh
    unzip individual_stocks_5yr.zip
    ./merge.sh
    cd ..

    echo "Finish downloading stock prediction data"
fi

python ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction.py
cd $cwd

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-stock-prediction failed"
    exit $exit_status
fi
now=$(date "+%s")
time6=$((now-start))
echo "#6 chronos-stock-prediction time used:$time6 seconds"

echo "#7 start app test for chronos-network-traffic-multivarite-multistep-tcnforecaster"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster

FILENAME="${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists."
else
    echo "Downloading network traffic data"

    wget $FTP_URI/analytics-zoo-data/network_traffic/data/data.csv -P ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data

    echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
sed -i "s/epochs=20/epochs=1/g" ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/

python ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-multivariate-multistep-tcnforecaster failed"
    exit $exit_status
fi

now=$(date "+%s")
time7=$((now-start))
echo "#7 chronos-network-traffic-multivarite-multistep-tcnforecaster time used:$time7 seconds"

echo "#8 start app test for chronos-stock-prediction-prophet"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet

sed -i '/get_ipython()/d; /plot./d; /plt./d' ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet.py
sed -i "s/epochs\ =\ 50/epochs\ =\ 2/g; s/batch_size\ =\ 16/batch_size\ =\ 108/g" ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet.py
cwd=$PWD
cd ${BIGDL_ROOT}/python/chronos/use-case/fsi/

# download data
if [ -d "data" ]
then
    echo "data already exists"
else
    echo "Downloading stock prediction data"

    mkdir data
    cd data
    wget https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip
    wget https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/merge.sh
    chmod +x merge.sh
    unzip individual_stocks_5yr.zip
    ./merge.sh
    cd ..

    echo "Finish downloading stock prediction data"
fi

python ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet.py
cd $cwd

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-stock-prediction-prophet failed"
    exit $exit_status
fi
now=$(date "+%s")
time8=$((now-start))
echo "#8 chronos-stock-prediction-prophet time used:$time8 seconds"

echo "#9 start app test for chronos-network-traffic-autots-forecasting"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting

FILENAME="${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
sed -i "s/cores=10/cores=4/g; s/epochs=20/epochs=1/g; s/n_sampling=4/n_sampling=1/g" ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/

python ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-autots-forecasting failed"
    exit $exit_status
fi
now=$(date "+%s")
time9=$((now-start))
echo "#9 chronos-network-traffic-autots-forecasting time used:$time9 seconds"

echo "#10 start app test for chronos-network-traffic-autots-customized-model"
#timer
time10=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.py
sed -i 's/epochs=5/epochs=1/g;s/choice([32,64])/choice([32])/g;s/grid_search([32, 64])/grid_search([2, 4])/g' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.py

if [ -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]
then
    echo "network_traffic_data.csv exists."
else

    echo "Download network traffic data."    
    wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/

    cd ~/.chronos/dataset/network_traffic/
    mv data.csv network_traffic_data.csv

    echo "Finished downloading network_traffic_data.csv"

    cd -
fi

python ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network_traffic_autots_customized_model failed."
    exit $exit_status
fi
now=$(date "+%s")
time10=$((now-start))
echo "#10 network_traffic_autots_customized_model time used:$time10 seconds"

# This should be done at the very end after all tests finish.
clear_up
