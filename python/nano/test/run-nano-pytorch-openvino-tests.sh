#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_OPENVINO_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/openvino/pytorch
# temporary fix on torchmetrics, downgrade to 0.7.2
pip install torchmetrics==0.7.2 --upgrade

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${PYTORCH_OPENVINO_NANO_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"
