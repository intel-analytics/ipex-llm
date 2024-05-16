#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain_gpu

if [[ $1 == "--win-arc" ]]; then
  export SYCL_CACHE_PERSISTENT=1
else
  export USE_XETLA=OFF
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi

export DEVICE='xpu'

set -e

echo "# Start testing inference"
start=$(date "+%s")

python -m pytest -s ${LLM_INFERENCE_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm langchain gpu tests finished"
echo "Time used:$time seconds"