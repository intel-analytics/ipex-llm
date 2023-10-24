source bigdl-llm-init
export MASTER_ADDR=127.0.0.1
export CCL_ZE_IPC_EXCHANGE=sockets
if [[ -n $OMP_NUM_THREADS ]]; then
    export OMP_NUM_THREADS=$(($OMP_NUM_THREADS / 4))
else
    export OMP_NUM_THREADS=$(($(nproc) / 4))
fi
torchrun --standalone \
         --nnodes=1 \
         --nproc-per-node 4 \
         deepspeed_autotp.py --repo-id-or-model-path "meta-llama/Llama-2-7b-hf"
