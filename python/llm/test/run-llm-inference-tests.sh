#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/inference

set -e

echo "# Start testing inference"
start=$(date "+%s")

echo '=====TEST====='
echo ${LLM_INFERENCE_TEST_DIR}/test_call_models.py
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/test_call_models.py -v

if [ -z "$THREAD_NUM" ]; then
  THREAD_NUM=2
fi
export OMP_NUM_THREADS=$THREAD_NUM
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/test_transformers_api.py -v
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/test_optimize_model_api.py -v

python -m pip install transformers==4.34.0
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/test_transformesr_api_434.py -v
python -m pip install transformers==4.31.0

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"
