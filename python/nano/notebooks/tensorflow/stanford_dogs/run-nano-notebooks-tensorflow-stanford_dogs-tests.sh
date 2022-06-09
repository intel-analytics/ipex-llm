#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export TENSORFLOW_NANO_NOTEBOOKS_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/tensorflow/stanford_dogs
export FIT_STEPS=2
export VAL_STEPS=2
export FREEZE_EPOCHS=2
export UNFREEZE_EPOCHS=1
export BATCH_SIZE=8
export TEST_STEPS=128

set -e

echo "# Start Testing stanford_dogs Fit Notebook"
start=$(date "+%s")

python -m pytest --nbmake --nbmake-timeout=300 --nbmake-kernel=python3 ${TENSORFLOW_NANO_NOTEBOOKS_DIR}/nano_tensorflow_fit_example.ipynb

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano cifar10 train notebook test finished"
echo "Time used:$time seconds"

echo "# Start Testing stanford_dogs Inference Notebook"
start=$(date "+%s")

jupyter nbconvert --to notebook --ExecutePreprocessor.kernel_name python3 --execute ${TENSORFLOW_NANO_NOTEBOOKS_DIR}/nano_tensorflow_inference_example.ipynb

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-nano cifar10 train notebook test finished"
echo "Time used:$time seconds"

