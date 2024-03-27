source /opt/intel/oneapi/setvars.sh
 
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export ENABLE_SDP_FUSION=1
export SYCL_CACHE_PERSISTENT=1
KERNEL_VERSION=$(uname -r)
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
 
python run.py # make sure config YAML file