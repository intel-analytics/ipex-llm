# need to update transformers version first
# pip install transformers==4.37.0
cd /llm/lightweight_serving
export IPEX_LLM_NOT_USE_VLLM=True
model_path="/llm/models/Llama-2-7b-chat-hf"
low_bit="sym_int4"
python lightweight_serving.py --repo-id-or-model-path $model_path --low-bit $low_bit