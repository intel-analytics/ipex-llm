#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export PYTORCH_NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test/pytorch

wget -nv ${FTP_URI}/analytics-zoo-data/cifar-10-python.tar.gz -P ${PYTORCH_NANO_TEST_DIR}/data

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${PYTORCH_NANO_TEST_DIR}/tests/ -k 'not ipex'

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"

# Install ipex to test IPEX related modules
pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable
echo "# Start testing IPEX"
start=$(date "+%s")
python -m pytest -s ${PYTORCH_NANO_TEST_DIR}/tests/ -k 'ipex'

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests finished"
echo "Time used:$time seconds"