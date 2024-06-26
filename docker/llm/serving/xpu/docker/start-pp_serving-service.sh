source /opt/intel/oneapi/setvars.sh
export no_proxy=localhost
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=32

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

export USE_XETLA=OFF
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export TORCH_LLM_ALLREDUCE=0

export NUM_GPUS=2
export IPEX_LLM_QUANTIZE_KV_CACHE=1

export MODEL_PATH="/llm/models/Llama-2-7b-chat-hf"
export low_bit="fp8"
# max requests = max_num * rank_num
export max_num="4"
cd /llm/pp_serving
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS pipeline_serving.py --repo-id-or-model-path $MODEL_PATH --low-bit $low_bit --max-num-seqs $max_num