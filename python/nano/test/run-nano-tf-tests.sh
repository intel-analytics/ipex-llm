#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export TF_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/tf

set -e
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${TF_NANO_TEST_DIR} -k "not test_graph_mode_fit"

python -m pytest -s ${TF_NANO_TEST_DIR} -k "test_graph_mode_fit"

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"

