#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/inference

set -e

python -m pip install transformers==4.36.2

echo "# Start testing inference"
start=$(date "+%s")

python -m pytest -s ${LLM_INFERENCE_TEST_DIR} -k "not test_transformers" -v

if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM
python -m pytest -s ${LLM_INFERENCE_TEST_DIR} -k test_transformers -v \
      --ignore=${LLM_INFERENCE_TEST_DIR}/test_optimize_mistral.py

python -m pip install transformers==4.31.0

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"
