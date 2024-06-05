#!/bin/bash
model="YOUR_MODEL_PATH"
served_model_name="YOUR_MODEL_NAME"
 
 
python -m ipex_llm.vllm.cpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --device cpu \
  --dtype bfloat16 \
  --enforce-eager \
  --load-in-low-bit bf16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 1