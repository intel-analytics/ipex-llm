# init llama-cpp first
mkdir -p /llm/llama-cpp
cd /llm/llama-cpp
init-llama-cpp

# change the model_path to run
model="/models/mistral-7b-v0.1.Q4_0.gguf"
./llama-cli -m $model -n 32 --prompt "What is AI?" -t 8 -e -ngl 999 --color
