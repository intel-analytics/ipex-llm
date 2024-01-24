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

if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM
pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py -v -s
export BIGDL_LLM_XMX_DISABLED=1
pytest ${LLM_INFERENCE_TEST_DIR}/test_transformers_api_disable_xmx.py -v -s
unset BIGDL_LLM_XMX_DISABLED

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu inference tests finished"
echo "Time used:$time seconds"

echo "# Start testing layers.fast_rope_embedding"
start=$(date "+%s")

pytest ${LLM_INFERENCE_TEST_DIR}/test_layer_fast_rope.py -v -s

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm gpu layers.fast_rope_embedding tests finished"
echo "Time used:$time seconds"
