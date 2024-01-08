#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/inference_gpu

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export DEVICE='xpu'

set -e

echo "# Start testing inference"
start=$(date "+%s")

python -m pip install transformers==4.36.2

if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM
#pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py -v -s
pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py::test_completion[llama] -v -s
pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py::test_optimize_model[llama] -v -s

python -m pip install transformers==4.31.0

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu tests finished"
echo "Time used:$time seconds"
