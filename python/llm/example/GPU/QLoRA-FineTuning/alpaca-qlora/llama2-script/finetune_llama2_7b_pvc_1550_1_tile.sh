python ./alpaca_qlora_finetuning.py \
    --base_model /home/wangruonan/zcg/Llama-2-7b-chat-hf \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qlora-alpaca" \
    --micro_batch_size 16 \
    --batch_size 128 \
    --gradient_checkpointing True
