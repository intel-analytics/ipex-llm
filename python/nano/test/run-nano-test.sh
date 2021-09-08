#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${NANO_TEST_DIR}/data

set -e
echo "#Start bigdl-nano tests"
echo "#1 Start test imagefolder"
start=$(date "+%s")
python -m pytest -s ${NANO_TEST_DIR}/test_imagefolder.py
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start test models vision"
start=$(date "+%s")
python -m pytest -s ${NANO_TEST_DIR}/test_models_vision.py
now=$(date "+%s")
time2=$((now-start))

echo "Bigdl-nano tests finished"
echo "#1 imagefolder time used:$time1 seconds"
echo "#2 models vision time used:$time2 seconds"
