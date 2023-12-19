python alpaca_qlora_finetuning.py \
    --micro_batch_size 8 \
    --batch_size 128 \
    --num_epochs 3 \
    --gradient_checkpointing True \
    --base_model 'tiiuae/falcon-40b'  \
    --lora_target_modules "["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]"
