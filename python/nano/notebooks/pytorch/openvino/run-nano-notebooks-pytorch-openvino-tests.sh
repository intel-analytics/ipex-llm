#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_OPENVINO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/pytorch/openvino
export DEV_RUN=True


set -e

echo "# Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_OPENVINO_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests for OpenVINO finished"
echo "Time used:$time seconds"