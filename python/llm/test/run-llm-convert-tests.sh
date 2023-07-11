#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_CONVERT_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/convert

set -e

echo "# Start testing convert"
start=$(date "+%s")

# separate convert process to save disk space
if [[ $1 == "llama" ]]; then
  python -m pytest -s ${LLM_CONVERT_TEST_DIR}/test_convert_model.py -k "test_convert_llama"
elif [[ $1 == "gptneox" ]]; then
  python -m pytest -s ${LLM_CONVERT_TEST_DIR}/test_convert_model.py -k "test_convert_gptneox"
elif [[ $1 == "bloom" ]]; then
  python -m pytest -s ${LLM_CONVERT_TEST_DIR}/test_convert_model.py -k "test_convert_bloom"
elif [[ $1 == "starcoder" ]]; then
  python -m pytest -s ${LLM_CONVERT_TEST_DIR}/test_convert_model.py -k "test_convert_starcoder"
fi

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"
