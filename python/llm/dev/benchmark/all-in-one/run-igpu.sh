source /opt/intel/oneapi/setvars.sh
 
export SYCL_CACHE_PERSISTENT=1
export BIGDL_LLM_XMX_DISABLED=1
 
python run.py # make sure config YAML file
