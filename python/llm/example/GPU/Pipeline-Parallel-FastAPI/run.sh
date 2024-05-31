source /opt/intel/oneapi/setvars.sh
export no_proxy=localhost
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=8

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

export MODEL_PATH=YOUR_MODEL_PATH
export NUM_GPUS=2
CCL_ZE_IPC_EXCHANGE=sockets torchrun --standalone --nnodes=1 --nproc-per-node $NUM_GPUS pipeline_serving.py --repo-id-or-model-path $MODEL_PATH --low-bit fp8
