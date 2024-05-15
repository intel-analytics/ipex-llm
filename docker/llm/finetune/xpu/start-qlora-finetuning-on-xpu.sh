#!/bin/bash
set -x
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
source /opt/intel/oneapi/setvars.sh

if [ -d "./model" ];
then
  MODEL_PARAM="--repo-id-or-model-path ./model"  # otherwise, default to download from HF repo
fi

if [ -d "./data/alpaca-cleaned" ];
then
  DATA_PARAM="--dataset ./data/alpaca-cleaned" # otherwise, default to download from HF dataset
fi

# QLoRA example dir
cd /LLM-Finetuning/QLoRA/simple-example/

python qlora_finetuning.py $MODEL_PARAM $DATA_PARAM
