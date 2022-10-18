#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_NANO_NOTEBOOKS_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/pytorch/cifar10
export DEV_RUN=True
export SUBSET=50

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${PYTORCH_NANO_NOTEBOOKS_DIR}/

set -e

echo "# Start Testing cifar10 train notebook"
start=$(date "+%s")

python -m pytest --nbmake --nbmake-timeout=180 --nbmake-kernel=python3 ${PYTORCH_NANO_NOTEBOOKS_DIR}/nano-trainer-example.ipynb

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano cifar10 train notebook test finished"
echo "Time used:$time seconds"

echo "# Start Testing cifar10 inference notebook"
start=$(date "+%s")

python -m pytest --nbmake --nbmake-timeout=180 --nbmake-kernel=python3 ${PYTORCH_NANO_NOTEBOOKS_DIR}/nano-inference-example.ipynb

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano cifar10 inference notebook tests finished"
echo "Time used:$time seconds"
