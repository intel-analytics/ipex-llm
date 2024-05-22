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

# pytest may return exit code 127, which cause unexpected error
# ref: https://github.com/intel/intel-extension-for-pytorch/issues/634
pytest_check_error() {
  result=$(eval "$@" || echo "FINISH PYTEST")
  echo $result > pytest_check_error.log
  cat pytest_check_error.log
  failed_lines=$(cat pytest_check_error.log | { grep failed || true; })
  if [[ $failed_lines != "" ]]; then
    exit 1
  fi
  rm pytest_check_error.log
}

export BIGDL_LLM_XMX_DISABLED=1
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_attention.py -v -s -k "Mistral"
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_mlp.py -v -s -k "Mistral"
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_RMSNorm.py -v -s -k "Mistral"
unset BIGDL_LLM_XMX_DISABLED

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu inference tests for transformers 4.34.0 finished"
echo "Time used:$time seconds"
