#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/inference/openvino

set -e

# download model
omz_downloader --name resnet18-xnor-binary-onnx-0001 -o ./model

# comment out the install commands
sed -i 's/!pip install/#!pip install/' $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb

# comment out the environment setting commands
sed -i 's/!source bigdl-nano-init/#!source bigdl-nano-init/' $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb

echo 'Start testing'
start=$(date "+%s")

# skip accelerate_inference_openvino_gpu.ipynb for now 
# as GPU is not supported in Nano action tests
python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR} -k 'not accelerate_inference_openvino_gpu'

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Infernece OpenVINO finished."
echo "Time used: $time seconds"
