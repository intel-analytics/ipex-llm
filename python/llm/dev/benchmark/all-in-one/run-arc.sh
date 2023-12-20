source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export BIGDL_IMPORT_IPEX="True"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python run.py # make sure config YAML file
