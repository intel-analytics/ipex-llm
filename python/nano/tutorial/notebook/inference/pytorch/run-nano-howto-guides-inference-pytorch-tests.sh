#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/inference/pytorch

set -e

# the number of batches to run is limited for testing purposes
sed -i "s/Trainer(max_epochs=1)/Trainer(max_epochs=1, fast_dev_run=True)/" $NANO_HOWTO_GUIDES_TEST_DIR/quantize_pytorch_inference_inc.ipynb $NANO_HOWTO_GUIDES_TEST_DIR/quantize_pytorch_inference_pot.ipynb $NANO_HOWTO_GUIDES_TEST_DIR/inference_optimizer_optimize.ipynb

# reduce the executing time of optimize function for testing purposes
sed -i "s/latency_sample_num=100/latency_sample_num=10/" $NANO_HOWTO_GUIDES_TEST_DIR/inference_optimizer_optimize.ipynb
sed -i "s/latency_sample_num=20/latency_sample_num=10/" $NANO_HOWTO_GUIDES_TEST_DIR/inference_optimizer_optimize.ipynb

# It seems windows's bash cannot expand * wildcard
all_ipynb=`find "$NANO_HOWTO_GUIDES_TEST_DIR" -maxdepth 1 -name "*.ipynb"`

# comment out the install commands
sed -i "s/!pip install/#!pip install/" $all_ipynb

# comment out the environment setting commands
sed -i "s/!source bigdl-nano-init/#!source bigdl-nano-init/" $all_ipynb

echo "Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR} -k "not inference_optimizer_optimize"
python -m pytest -s --nbmake --nbmake-timeout=1800 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR}/inference_optimizer_optimize.ipynb

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Inference PyTorch finished."
echo "Time used: $time seconds"
