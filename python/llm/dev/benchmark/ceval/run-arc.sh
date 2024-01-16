source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python eval.py \
    --model_family llama \
    --model_path "path to model" \
    --eval_type validation \
    --device xpu \
    --eval_data_path data \
    --qtype sym_int4