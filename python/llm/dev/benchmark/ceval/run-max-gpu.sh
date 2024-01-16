source /opt/intel/oneapi/setvars.sh
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1

python eval.py \
    --model_family llama \
    --model_path "path to model" \
    --eval_type validation \
    --device xpu \
    --eval_data_path data \
    --qtype sym_int4