mkdir -p ~/ollama
cd ~/ollama
init-ollama
export OLLAMA_NUM_GPU=999
export ZES_ENABLE_SYSMAN=1
source /opt/intel/oneapi/setvars.sh --force
export SYCL_CACHE_PERSISTENT=1
KERNEL_VERSION=$(uname -r)
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi

(./ollama serve > ollama.log) &
