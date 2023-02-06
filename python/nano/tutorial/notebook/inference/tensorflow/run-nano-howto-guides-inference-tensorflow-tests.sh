#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_HOWTO_GUIDES_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/tutorial/notebook/inference/tensorflow

AVX512_AVAILABLE=`lscpu | grep avx512`

set -e

# comment out the install commands
sed -i "s/!pip install/#!pip install/" $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb

# comment out the environment setting commands
sed -i "s/!source bigdl-nano-init/#!source bigdl-nano-init/" $NANO_HOWTO_GUIDES_TEST_DIR/*.ipynb

# limit the iterations of inferece for testing purposes
sed -i "s/range(100)/range(10)/" $NANO_HOWTO_GUIDES_TEST_DIR/tensorflow_inference_bf16.ipynb

echo "Start testing"
start=$(date "+%s")

python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR} -k "not tensorflow_inference_bf16"
if [[ ! -z "$AVX512_AVAILABLE" ]]; then
    python -m pytest -s --nbmake --nbmake-timeout=600 --nbmake-kernel=python3 ${NANO_HOWTO_GUIDES_TEST_DIR}/tensorflow_inference_bf16.ipynb
fi

now=$(date "+%s")
time=$((now-start))

echo "BigDL-Nano how-to guides tests for Infernece TensorFLow finished."
echo "Time used: $time seconds"
