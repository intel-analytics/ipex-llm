#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export OPENVINO_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/openvino/basic

# Download openvino model
omz_downloader --name resnet18-xnor-binary-onnx-0001

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${OPENVINO_NANO_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"
