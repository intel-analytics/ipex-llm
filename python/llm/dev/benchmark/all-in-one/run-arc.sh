source /opt/intel/oneapi/setvars.sh
 
export USE_XETLA=OFF
export SYCL_CACHE_PERSISTENT=1
KERNEL_VERSION=$(uname -r)
if [[ $KERNEL_VERSION != *"6.5"* ]]; then
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
fi
 
python run.py # make sure config YAML file