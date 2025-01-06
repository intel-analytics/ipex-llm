#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/inference_gpu

if [[ $RUNNER_OS == "Linux" ]]; then
  export USE_XETLA=OFF
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
elif [[ $RUNNER_OS == "Windows" ]]; then
  export ANALYTICS_ZOO_ROOT=$(cygpath -m ${ANALYTICS_ZOO_ROOT})
  export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/inference_gpu
  export SYCL_CACHE_PERSISTENT=1
fi

export DEVICE='xpu'

set -e

echo "# Start testing inference"
start=$(date "+%s")

# if [ -z "$THREAD_NUM" ]; then
#   THREAD_NUM=2
# fi
# export OMP_NUM_THREADS=$THREAD_NUM

# import pytest_check_error function
source ${ANALYTICS_ZOO_ROOT}/python/llm/test/run-llm-check-function.sh

pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py -v -s
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_final_logits.py -v -s
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_attention.py -v -s
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_mlp.py -v -s
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_RMSNorm.py -v -s

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu inference tests finished"
echo "Time used:$time seconds"
