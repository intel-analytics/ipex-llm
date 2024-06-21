source /opt/intel/oneapi/setvars.sh
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9090
export FI_PROVIDER=tcp
export USE_XETLA=OFF
export OMP_NUM_THREADS=6
export IPEX_LLM_QUANTIZE_KV_CACHE=1
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export TORCH_LLM_ALLREDUCE=0

NUM_GPUS=2 # number of used GPU

# To run Yi-6B-Chat
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
    generate.py --repo-id-or-model-path '01-ai/Yi-6B-Chat' --gpu-num $NUM_GPUS

# To run Yi-34B-Chat
# CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
#     generate.py --repo-id-or-model-path '01-ai/Yi-34B-Chat' --gpu-num $NUM_GPUS
