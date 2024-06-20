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

# To run Mistral-7B-v0.1
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS \
    generate.py --repo-id-or-model-path 'mistralai/Mistral-7B-v0.1' --gpu-num $NUM_GPUS

# Todo: Mixtral-8x7B-Instruct-v0.1
