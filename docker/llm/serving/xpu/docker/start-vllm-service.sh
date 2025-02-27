#!/bin/bash
MODEL_PATH=${MODEL_PATH:-"default_model_path"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"default_model_name"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}  # Default to 1 if not set

echo "Starting service with model: $MODEL_PATH"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"

export CCL_WORKER_COUNT=2
export SYCL_CACHE_PERSISTENT=1
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
 
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

export CCL_SAME_STREAM=1
export CCL_BLOCKING_WAIT=0
 
source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $SERVED_MODEL_NAME \
  --port 8000 \
  --model $MODEL_PATH \
  --trust-remote-code \
  --block-size 8 \
  --gpu-memory-utilization 0.95 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 2000 \
  --max-num-batched-tokens 3000 \
  --max-num-seqs 256 \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --disable-async-output-proc \
  --distributed-executor-backend ray
