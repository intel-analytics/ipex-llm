#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export INC_TF_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/onnx/tf

set -e
echo "# Start testing"
start=$(date "+%s")
# It seems nano's default `MALLOC_CONF` will cause higher memory usage,
# and cause OOM (Killed) in git action
unset MALLOC_CONF
python -m pytest -s ${INC_TF_NANO_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"

