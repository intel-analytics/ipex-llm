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
  echo HERE 1
  echo $result > pytest_check_error.log
  echo HERE 2
  cat pytest_check_error.log
  echo HERE 3
  failed_lines=$(cat pytest_check_error.log | { grep failed || true; })
  echo HERE 4
  if [[ $failed_lines != "" ]]; then
    echo HERE 5
    exit 1
  fi
  echo HERE 6
  rm pytest_check_error.log
  echo HERE 7
}

pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py -v -s
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_layernorm.py -v -s

export BIGDL_LLM_XMX_DISABLED=1

pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_final_logits.py -v -s
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_attention.py -v -s -k "not Mistral"
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_mlp.py -v -s -k "not Mistral"
pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_RMSNorm.py -v -s -k "not Mistral"

unset BIGDL_LLM_XMX_DISABLED

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu inference tests finished"
echo "Time used:$time seconds"

echo "# Start testing layers.fast_rope_embedding"
start=$(date "+%s")

pytest_check_error pytest ${LLM_INFERENCE_TEST_DIR}/test_layer_fast_rope.py -v -s

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu layers.fast_rope_embedding tests finished"
echo "Time used:$time seconds"
