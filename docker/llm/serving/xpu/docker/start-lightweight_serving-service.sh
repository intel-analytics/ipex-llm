cd /llm/lightweight_serving
model_path="/llm/models/Llama-2-7b-chat-hf"
low_bit="sym_int4"
source /opt/intel/1ccl-wks/setvars.sh
python lightweight_serving.py --repo-id-or-model-path $model_path --low-bit $low_bit