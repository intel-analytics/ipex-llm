#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/pytorch

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${PYTORCH_NANO_TEST_DIR}/tests/test_inference_optimizer.py

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"
