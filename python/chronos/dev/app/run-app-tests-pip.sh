#!/usr/bin/env bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

clear_up () {
    echo "Clearing up environment. Uninstalling bigdl"
    pip uninstall -y bigdl-chronos
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

chmod +x ${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh

execute_test(){
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
    echo "$1 finished."
    return $((now-start))
}


# ${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based
FILENAME="${BIGDL_ROOT}/python/chronos/use-case/AIOps/m_1932.csv"
if [ ! -f "$FILENAME" ];then
   wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -O ${BIGDL_ROOT}/python/chronos/use-case/AIOps/m_1932.csv
fi

# sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
# sed -i "s/epochs=20/epochs=1/g" ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
# cd ${BIGDL_ROOT}/python/chronos/use-case/AIOps/
# execute_test anomaly-detect-unsupervised-forecast-based "${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py"
# time1=$?
# cd -

${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised
sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.py
cd ${BIGDL_ROOT}/python/chronos/use-case/AIOps/
execute_test anomaly-detect-unsupervised "${BIGDL_ROOT}/python/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.py"
time2=$?
cd -


if [ ! -f ~/.chronos/dataset/electricity.csv ];then
    wget -nv $FTP_URI/analytics-zoo-data/apps/electricity.csv -P ~/.chronos/dataset/
fi
sed -i "s/epochs=30/epochs=1/g" ${BIGDL_ROOT}/python/chronos/use-case/electricity/tcn.py
execute_test electricity-tcn "${BIGDL_ROOT}/python/chronos/use-case/electricity/tcn.py"
time3=$?

# sed -i "s/epochs=3/epochs=1/g" ${BIGDL_ROOT}/python/chronos/use-case/electricity/autoformer.py
# execute_test electricity-autoformer "${BIGDL_ROOT}/python/chronos/use-case/electricity/autoformer.py"
# time4=$?


${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction
cd ${BIGDL_ROOT}/python/chronos/use-case/fsi/
if [ ! -d "data" ];then
    mkdir data
    cd data
    wget https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip
    wget https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/merge.sh
    chmod +x merge.sh
    unzip individual_stocks_5yr.zip
    ./merge.sh
    cd ..
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction.py
sed -i "s/epochs\ =\ 50/epochs\ =\ 1/g; s/batch_size\ =\ 16/batch_size\ =\ 108/g" ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction.py
execute_test stock-prediction "${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction.py"
time5=$?
cd -

${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet
sed -i '/get_ipython()/d; /plot./d; /plt./d' ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet.py
sed -i "s/epochs\ =\ 50/epochs\ =\ 1/g; s/batch_size\ =\ 16/batch_size\ =\ 108/g" ${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet.py
cd ${BIGDL_ROOT}/python/chronos/use-case/fsi/
execute_test stock-prediction-prophet "${BIGDL_ROOT}/python/chronos/use-case/fsi/stock_prediction_prophet.py"
time6=$?
cd -


# ${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting
FILENAME="${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data/data.csv"
if [ ! -f "$FILENAME" ];then
   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data
fi

# sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
# sed -i "s/epochs=50/epochs=1/g; s/epochs=20/epochs=1/g" ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
# cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/
# execute_test network-traffic-model-forecasting "${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_model_forecasting.py"
# time7=$?
# cd -

${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster
sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
sed -i "s/epochs=20/epochs=1/g" ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/
execute_test network-traffic-multivariate-multistep-tcnforecaster "${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py"
time8=$?
cd -

${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting
sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
sed -i "s/cores=10/cores=4/g; s/epochs=20/epochs=1/g; s/n_sampling=24/n_sampling=6/g" ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/
execute_test network-traffic-autots-forecasting "${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py"
time9=$?
cd -

${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model
sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.py
sed -i 's/epochs=5/epochs=1/g;s/choice([32,64])/choice([32])/g;s/grid_search([32, 64])/grid_search([2, 4])/g' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.py
if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ];then
    wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
    mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi
execute_test network-traffic-autots-customized-model "${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_customized_model.py"
time10=$?


# Because of the long time cost, test 1/4/7 temporarily skipped
# echo "#1 anomaly-detect-unsupervised-forecast-based time used:$time1 seconds"
echo "#2 anomaly-detect-unsupervised time used:$time2 seconds"
echo "#3 electricity-tcn time used:$time3 seconds"
# echo "#4 electricity-autoformer time used:$time4 seconds"
echo "#5 stock-prediction time used:$time5 seconds"
echo "#6 stock-prediction-prophet time used:$time6 seconds"
# echo "#7 network-traffic-model-forecasting time used:$time7 seconds"
echo "#8 network-traffic-multivarite-multistep-tcnforecaster time used:$time8 seconds"
echo "#9 network-traffic-autots-forecasting time used:$time9 seconds"
echo "#10 network-traffic-autots-customized-model time used:$time10 seconds"

# This should be done at the very end after all tests finish.
clear_up
