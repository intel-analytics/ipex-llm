#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export LLM_INFERENCE_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain_gpu

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export DEVICE='xpu'

set -e

echo "# Start testing inference"
start=$(date "+%s")

python -m pytest -s ${LLM_INFERENCE_TEST_DIR}

mkdir ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir
wget https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/community/tests/integration_tests/llms/test_bigdl.py -P ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir
sed -i "s,model_id=\"[^\"]*\",model_id=\"$LLAMA2_7B_ORIGIN_PATH\",g" ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir/test_bigdl.py
python -m pytest -s ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir
rm -rf ${LLM_INFERENCE_TEST_DIR}/tmp_wget_dir

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm langchain gpu tests finished"
echo "Time used:$time seconds"