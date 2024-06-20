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

# To run CodeLlama-7b-Instruct-hf
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
    generate.py --repo-id-or-model-path 'codellama/CodeLlama-7b-Instruct-hf' --gpu-num $NUM_GPUS

# To run CodeLlama-13b-Instruct-hf
# CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
#     generate.py --repo-id-or-model-path 'codellama/CodeLlama-7b-Instruct-hf' --gpu-num $NUM_GPUS

# To run CodeLlama-34b-Instruct-hf
# CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
#     generate.py --repo-id-or-model-path 'codellama/CodeLlama-34b-Instruct-hf' --gpu-num $NUM_GPUS
