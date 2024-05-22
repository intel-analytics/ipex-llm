#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}

set -e

echo "# Start testing qlora fine-tuning"
start=$(date "+%s")

sed -i 's/max_steps=200/max_steps=2/; s/save_steps=100/save_steps=2/; s/logging_steps=20/logging_steps=1/' \
    ${ANALYTICS_ZOO_ROOT}/python/llm/example/GPU/LLM-Finetuning/QLoRA/simple-example/qlora_finetuning.py

# ipex may return exit code 127, which cause unexpected error
# ref: https://github.com/intel/intel-extension-for-pytorch/issues/634
workaround_wrapper() {
    result=$(eval "$@" || ( [ $? -eq 127 && $RUNNER_OS == "Windows" ] && echo "EXIT CODE 127 DETECTED ON WINDOWS, IGNORE."))
    echo $result > example_test.log
    cat example_test.log
    rm example_test.log
}

workaround_wrapper python ${ANALYTICS_ZOO_ROOT}/python/llm/example/GPU/LLM-Finetuning/QLoRA/simple-example/qlora_finetuning.py \
--repo-id-or-model-path ${LLAMA2_7B_ORIGIN_PATH} \
--dataset ${YAHMA_ALPACA_CLEANED_PATH}

workaround_wrapper python ${ANALYTICS_ZOO_ROOT}/python/llm/example/GPU/LLM-Finetuning/QLoRA/simple-example/export_merged_model.py \
--repo-id-or-model-path ${LLAMA2_7B_ORIGIN_PATH} \
--adapter_path ${PWD}/outputs/checkpoint-2 \
--output_path ${PWD}/outputs/checkpoint-2-merged

now=$(date "+%s")
time=$((now-start))

echo "qlora fine-tuning test finished"
echo "Time used:$time seconds"
