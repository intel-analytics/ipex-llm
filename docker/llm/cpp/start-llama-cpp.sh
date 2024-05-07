mkdir -p ~/llama-cpp
cd ~/llama-cpp
init-llama-cpp
source /opt/intel/oneapi/setvars.sh --force
export SYCL_CACHE_PERSISTENT=1
KERNEL_VERSION=$(uname -r)
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi

model="/models/mistral-7b-v0.1.Q4_0.gguf"
./main -m $model -n 32 --prompt "what is AI?" -t 8 -e -ngl 33 --color