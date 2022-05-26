#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_NANO_NOTEBOOKS_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/pytorch

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${PYTORCH_NANO_NOTEBOOKS_DIR}/cifar10

set -e

echo "# Start Testing Pytorch Inference Notebooks"
start=$(date "+%s")

python -m pytest --nbmake --nbmake-timeout=1000 --nbmake-kernel=python3 ${PYTORCH_NANO_NOTEBOOKS_DIR}/cifar10/nano-inference-example.ipynb

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano Train Notebooks tests finished"
echo "Time used:$time seconds"
