#!/bin/bash
set -x

if [ -d "/bigdl/model" ];
then
  MODEL_PARAM="--repo-id-or-model-path /bigdl/model"  # otherwise, default to download from HF repo
fi

if [ -d "/bigdl/data/english_quotes" ];
then
  DATA_PARAM="--dataset /bigdl/data/english_quotes" # otherwise, default to download from HF dataset
fi

if [ "$STANDALONE_DOCKER" = "TRUE" ]
then
  export CONTAINER_IP=$(hostname -i)
  export CPU_CORES=$(nproc)
  source /opt/intel/oneapi/setvars.sh
  export CCL_WORKER_COUNT=$WORKER_COUNT_DOCKER
  export CCL_WORKER_AFFINITY=auto
  export MASTER_ADDR=$CONTAINER_IP
  mpirun \
     -n $CCL_WORKER_COUNT \
     -ppn $CCL_WORKER_COUNT \
     -genv OMP_NUM_THREADS=$((CPU_CORES / CCL_WORKER_COUNT)) \
     -genv KMP_AFFINITY="granularity=fine,none" \
     -genv KMP_BLOCKTIME=1 \
     -genv TF_ENABLE_ONEDNN_OPTS=1 \
     python /bigdl/qlora_finetuning_cpu.py $MODEL_PARAM $DATA_PARAM

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python /bigdl/qlora_finetuning_cpu.py $MODEL_PARAM $DATA_PARAM

