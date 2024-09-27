export IPEX_LLM_LAST_LM_HEAD=0

python eval.py \
    --model_path "path to model" \
    --eval_type validation \
    --device xpu \
    --eval_data_path data \
    --qtype sym_int4