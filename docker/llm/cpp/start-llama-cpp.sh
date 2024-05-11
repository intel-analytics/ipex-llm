# init llama-cpp first
mkdir -p /llm/llama-cpp
cd /llm/llama-cpp
init-llama-cpp

# change the model_path to run
model="/models/mistral-7b-v0.1.Q4_0.gguf"
./main -m $model -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun" -t 8 -e -ngl 33 --color
