#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/inference/tensorflow/

set -e

sed -i s/epochs = 20/epochs = 1/g $NANO_TUTORIAL_TEST_DIR/pytorch_quantization.py
python $NANO_TUTORIAL_TEST_DIR/cifar10.py
