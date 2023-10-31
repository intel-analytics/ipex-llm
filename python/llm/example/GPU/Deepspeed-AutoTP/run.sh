source bigdl-llm-init -t -g
export MASTER_ADDR=127.0.0.1
export CCL_ZE_IPC_EXCHANGE=sockets
NUM_GPUS=4
if [[ -n $OMP_NUM_THREADS ]]; then
    export OMP_NUM_THREADS=$(($OMP_NUM_THREADS / $NUM_GPUS))
else
    export OMP_NUM_THREADS=$(($(nproc) / $NUM_GPUS))
fi
torchrun --standalone \
         --nnodes=1 \
         --nproc-per-node $NUM_GPUS \
         deepspeed_autotp.py --repo-id-or-model-path "meta-llama/Llama-2-7b-hf"
