#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}

set -e

echo "# Start testing qlora fine-tuning"
start=$(date "+%s")

python ${ANALYTICS_ZOO_ROOT}/python/llm/example/GPU/QLoRA-FineTuning/qlora_finetuning.py \
--repo-id-or-model-path ${LLAMA2_7B_ORIGIN_PATH} \
--dataset ${ABIRATE_ENGLISH_QUOTES_PATH}

python ${ANALYTICS_ZOO_ROOT}/python/llm/example/GPU/QLoRA-FineTuning/export_merged_model.py \
--repo-id-or-model-path ${LLAMA2_7B_ORIGIN_PATH} \
--adapter_path ${PWD}/outputs/checkpoint-200 \
--output_path ${PWD}/outputs/checkpoint-200-merged


now=$(date "+%s")
time=$((now-start))

echo "qlora fine-tuning tests finished"
echo "Time used:$time seconds"
