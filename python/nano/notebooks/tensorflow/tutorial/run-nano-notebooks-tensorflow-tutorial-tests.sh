#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/tensorflow/tutorial
export epochs=1
export STEPS=4
export NUM_SHARDS=50 

set -e

echo "# Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=180 --nbmake-kernel=python3 ${NANO_TUTORIAL_TEST_DIR} -k 'not quantization'

# meet error using nbmake
jupyter nbconvert --ExecutePreprocessor.timeout=300 --to notebook --execute ${NANO_TUTORIAL_TEST_DIR}/tensorflow_quantization.ipynb

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano tests for tutorial finished"
echo "Time used:$time seconds"