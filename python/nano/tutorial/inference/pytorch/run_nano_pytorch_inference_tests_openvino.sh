#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TUTORIAL_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/inference/pytorch/

set -e

python $NANO_TUTORIAL_TEST_DIR/pytorch_inference_openvino.py

sed -i s/Trainer\(max_epochs=1\)/Trainer\(max_epochs=1,\ fast_dev_run=True\)/ $NANO_TUTORIAL_TEST_DIR/pytorch_quantization_openvino.py
python $NANO_TUTORIAL_TEST_DIR/pytorch_quantization_openvino.py

python $NANO_TUTORIAL_TEST_DIR/pytorch_save_and_load_openvino.py
