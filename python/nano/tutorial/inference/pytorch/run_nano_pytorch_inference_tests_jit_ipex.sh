#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/inference/pytorch/

set -e

python $NANO_TUTORIAL_TEST_DIR/pytorch_inference_jit_ipex.py

python $NANO_TUTORIAL_TEST_DIR/pytorch_save_and_load_jit.py

python $NANO_TUTORIAL_TEST_DIR/pytorch_save_and_load_ipex.py
