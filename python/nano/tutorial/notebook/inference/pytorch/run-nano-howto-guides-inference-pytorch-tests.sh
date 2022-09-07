#!/bin/bash

# export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
# export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/inference/pytorch

set -e

# the number of batches to run is limited for testing purposes
sed -i 's/Trainer(max_epochs=1)/Trainer(max_epochs=1, fast_dev_run=True)/' $NANO_HOWTO_GUIDES_TEST_DIR/quantize_pytorch_inference_inc.ipynb $NANO_HOWTO_GUIDES_TEST_DIR/quantize_pytorch_inference_pot.ipynb

# comment out the install command for bigdl-nano[pytorch]
sed -i 's/!pip install --pre --upgrade bigdl-nano\[pytorch]/#!pip install --pre --upgrade bigdl-nano[pytorch]/' $NANO_HOWTO_GUIDES_TEST_DIR/*

echo "Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Inference PyTorch finished."
echo "Time used: $time seconds"
