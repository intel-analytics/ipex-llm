#!/bin/bash
model="YOUR_MODEL_PATH"
served_model_name="YOUR_MODEL_NAME"
 
 
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