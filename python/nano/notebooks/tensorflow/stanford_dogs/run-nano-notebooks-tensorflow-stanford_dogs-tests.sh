#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export TENSORFLOW_NANO_NOTEBOOKS_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/notebooks/tensorflow/stanford_dogs
export FIT_STEPS=4
export VAL_STEPS=2
export FREEZE_EPOCHS=2
export UNFREEZE_EPOCHS=1
export TEST_STEPS=2

set -e

echo "# Start Testing stanford_dogs Fit Notebook"

python -m pytest --nbmake --nbmake-timeout=300 --nbmake-kernel=python3 ${TENSORFLOW_NANO_NOTEBOOKS_DIR}/nano_tensorflow_fit_example.ipynb

echo "# Start Testing stanford_dogs Inference Notebook"

jupyter nbconvert --to script ${TENSORFLOW_NANO_NOTEBOOKS_DIR}/nano_tensorflow_inference_example.ipynb
python ${TENSORFLOW_NANO_NOTEBOOKS_DIR}/nano_tensorflow_inference_example.py
