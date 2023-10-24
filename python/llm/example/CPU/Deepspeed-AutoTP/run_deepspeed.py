#/bin/bash
export WORLD_SIZE=1
export REPO_ID_OR_MODEL_PATH=/home/llm/models/Llama-2-7b-chat-hf
export OMP_NUM_THREADS=48

export MASTER_ADDR=127.0.0.1
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ROOT=/opt/intel/oneccl
export DS_ACCELERATOR="cpu"
export CCL_WORKER_AFFINITY=auto

source /opt/intel/oneccl/env/setvars.sh

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node ${WORLD_SIZE} \
     test_deepspeed.py \
         --repo-id-or-model-path ${REPO_ID_OR_MODEL_PATH}

