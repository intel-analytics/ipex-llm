source /opt/intel/oneapi/setvars.sh
 
export USE_XETLA=OFF
export SYCL_CACHE_PERSISTENT=1
 
python run.py # make sure config YAML file
