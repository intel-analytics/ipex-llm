#!/bin/bash
set -e

export HF_TOKEN=${HF_TOKEN}
export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export LLM_HOME=${ANALYTICS_ZOO_ROOT}/python/llm/src
export BLOOM_ORIGIN_PATH=${ANALYTICS_ZOO_ROOT}/models/bloom-560m
# export LLAMA_ORIGIN_PATH=${ANALYTICS_ZOO_ROOT}/models/llama-7b-hf
export GPTNEOX_ORIGIN_PATH=${ANALYTICS_ZOO_ROOT}/models/redpajama-3b
export INT4_CKPT_DIR=${ANALYTICS_ZOO_ROOT}/models/converted_models
# export LLAMA_INT4_CKPT_PATH=${INT4_CKPT_DIR}/bigdl_llm_llama_7b_q4_0.bin
export GPTNEOX_INT4_CKPT_PATH=${INT4_CKPT_DIR}/bigdl_llm_redpajama_q4_0.bin
export BLOOM_INT4_CKPT_PATH=${INT4_CKPT_DIR}/bigdl_llm_bloom_q4_0.bin

echo "# Download the models"
start=$(date "+%s")

echo ${ANALYTICS_ZOO_ROOT}
python ${ANALYTICS_ZOO_ROOT}/python/llm/test/win/download_from_huggingface.py

now=$(date "+%s")
time=$((now-start))
echo "Models downloaded in:$time seconds"

echo "# Start testing convert model"
start=$(date "+%s")

python -m pytest -s ${ANALYTICS_ZOO_ROOT}/python/llm/test/convert/test_convert_model.py -k 'test_convert_bloom'

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm convert model test finished"
echo "Time used:$time seconds"


echo "# Start testing inference"
start=$(date "+%s")

python -m pytest -s ${ANALYTICS_ZOO_ROOT}/python/llm/test/inference/test_call_models.py -k 'test_bloom_completion_success or test_bloom_completion_with_stream_success'

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm inference test finished"
echo "Time used:$time seconds"


echo "# Start testing langchain"
start=$(date "+%s")

python -m pytest -s ${ANALYTICS_ZOO_ROOT}/python/llm/test/langchain/test_langchain.py -k 'test_langchain_llm_bloom'

now=$(date "+%s")
time=$((now-start))

echo "Bigdl-llm langchain test finished"
echo "Time used:$time seconds"
