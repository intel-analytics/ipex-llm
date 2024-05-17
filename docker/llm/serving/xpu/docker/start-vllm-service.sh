#!/bin/bash
model="YOUR_MODEL_PATH"
served_model_name="YOUR_MODEL_NAME"


export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export TORCH_LLM_ALLREDUCE=0
export CCL_DG2_ALLREDUCE=1


# Tensor parallel related arguments:
export CCL_WORKER_COUNT=2
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

source /opt/intel/1ccl-wks/setvars.sh
 
 
python -m ipex_llm.vllm.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 1