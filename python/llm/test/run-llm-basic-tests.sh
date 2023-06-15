#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_BASIC_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/basic

set -e

echo "# Start testing"
start=$(date "+%s")

echo "test install"
python -m pytest -s ${LLM_BASIC_TEST_DIR}/install

# TODO: supports tests on windows
platform=$1
if [[ $1 != "windows" ]]; then
  echo "test convert model"
  python -m pytest -s ${LLM_BASIC_TEST_DIR}/convert
fi

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"