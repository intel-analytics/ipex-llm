#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_BASIC_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/basics

set -e

# ipex is not installed here. Any tests needs ipex should be moved to next pytest command.
echo "# Start testing"
start=$(date "+%s")
python -m pytest -s ${LLM_BASIC_TEST_DIR}

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm tests finished"
echo "Time used:$time seconds"