python ./alpaca_qlora_finetuning.py \
    --base_model "/home/ywang30/.cache/huggingface/hub/Llama-2-70b-chat-hf" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./bigdl-qlora-alpaca" \
    --micro_batch_size 32 \
    --batch_size 128 \
    --gradient_checkpointing True
