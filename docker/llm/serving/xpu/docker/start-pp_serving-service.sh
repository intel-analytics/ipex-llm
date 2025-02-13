# update transformers version first
# pip install transformers==4.37.0
source /opt/intel/oneapi/setvars.sh --force
export IPEX_LLM_NOT_USE_VLLM=True
export no_proxy=localhost
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=32

#export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
# source $basekit_root/ccl/latest/env/vars.sh --force
source /opt/intel/1ccl-wks/setvars.sh

export USE_XETLA=OFF
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
export TORCH_LLM_ALLREDUCE=0

export IPEX_LLM_LAST_LM_HEAD=1
export IPEX_LLM_QUANTIZE_KV_CACHE=1
export IPEX_LLM_LOW_MEM=1
export num_gpus=2
export model_path="/llm/models/Llama-2-7b-chat-hf"
export low_bit="fp8"
# max requests = max_num_reqs * rank_num
export max_num_seqs="4"
export max_prefilled_seqs="0"

cd /llm/pp_serving
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $num_gpus pipeline_serving.py --repo-id-or-model-path $model_path --low-bit $low_bit --max-num-seqs $max_num_seqs --max-prefilled-seqs $max_prefilled_seqs
