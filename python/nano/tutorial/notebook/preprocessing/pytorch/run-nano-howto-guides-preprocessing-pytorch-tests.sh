#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/preprocessing/pytorch

set -e

# It seems windows's bash cannot expand * wildcard
all_ipynb=`find "$NANO_HOWTO_GUIDES_TEST_DIR" -maxdepth 1 -name "*.ipynb"`

# comment out the install commands
sed -i "s/!pip install/#!pip install/" $all_ipynb

# comment out the environment setting commands
sed -i "s/!source bigdl-nano-init/#!source bigdl-nano-init/" $all_ipynb

echo "Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Training PyTorch finished."
echo "Time used: $time seconds"
