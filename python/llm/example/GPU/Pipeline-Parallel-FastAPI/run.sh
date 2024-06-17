source /opt/intel/oneapi/setvars.sh
export no_proxy=localhost
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=32

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
basekit_root=/opt/intel/oneapi
source $basekit_root/setvars.sh --force
source $basekit_root/ccl/latest/env/vars.sh --force

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

export MODEL_PATH=YOUR_MODEL_PATH
export NUM_GPUS=2
export IPEX_LLM_QUANTIZE_KV_CACHE=1

CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS pipeline_serving.py --repo-id-or-model-path $MODEL_PATH --low-bit fp8 --max-num-seqs 4
