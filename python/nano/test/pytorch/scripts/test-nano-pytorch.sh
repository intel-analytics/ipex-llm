#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${PYTORCH_NANO_TEST_DIR}/data

set -e
echo "#Start bigdl-nano ipex tests"
echo "#1 Start test model vision ipex"

echo "#3 Start test models onnx"
start=$(date "+%s")
python -m pytest -s ${PYTORCH_NANO_TEST_DIR}/pytorch/test_lightning.py
now=$(date "+%s")
time3=$((now-start))

echo "Bigdl-nano ipex tests finished"
echo "#1 model vision ipex time used:$time1 seconds"
echo "#2 trainer ipex time used:$time2 seconds"
echo "#3 models onnx time used:$time3 seconds"
