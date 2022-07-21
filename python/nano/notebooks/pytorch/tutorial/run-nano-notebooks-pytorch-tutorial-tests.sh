#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/pytorch/tutorial
export DEV_RUN=True

openvino=$1

set -e

echo "# Start testing"
start=$(date "+%s")

if [ ${openvino} == true ]; then
    python -m pytest -s --nbmake --nbmake-timeout=180 --nbmake-kernel=python3 ${NANO_TUTORIAL_TEST_DIR} -k 'openvino'
else
    python -m pytest -s --nbmake --nbmake-timeout=180 --nbmake-kernel=python3 ${NANO_TUTORIAL_TEST_DIR} -k 'not openvino'
fi 

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests for tutorial finished"
echo "Time used:$time seconds"