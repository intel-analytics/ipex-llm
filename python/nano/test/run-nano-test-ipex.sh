#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${NANO_TEST_DIR}/data

source bigdl-nano-init

set -e
echo "#Start bigdl-nano ipex tests"
echo "#1 Start test model vision ipex"
start=$(date "+%s")
python -m pytest -s ${NANO_TEST_DIR}/test_models_vision_ipex.py
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start test trainer ipex"
start=$(date "+%s")
python -m pytest -s ${NANO_TEST_DIR}/test_trainer_ipex.py
now=$(date "+%s")
time2=$((now-start))

echo "#3 Start test models onnx"
start=$(date "+%s")
python -m pytest -s ${NANO_TEST_DIR}/test_models_onnx.py
now=$(date "+%s")
time3=$((now-start))

echo "Bigdl-nano ipex tests finished"
echo "#1 model vision ipex time used:$time1 seconds"
echo "#2 trainer ipex time used:$time2 seconds"
echo "#3 models onnx time used:$time3 seconds"
