#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/inference

source bigdl-nano-init

set -e

echo "# Start testing inference"
start=$(date "+%s")

# python -m pytest -s ${LLM_INFERENCE_TEST_DIR} --k "^test_transformers_int4"
export OMP_NUM_THREADS=24
taskset -c 0-23 python -m pytest -s ${LLM_INFERENCE_TEST_DIR} -k test_transformers_int4

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"
