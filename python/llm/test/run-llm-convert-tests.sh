#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_CONVERT_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/convert

set -e

echo "# Start testing convert"
start=$(date "+%s")

python -m pytest -s ${LLM_CONVERT_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"
