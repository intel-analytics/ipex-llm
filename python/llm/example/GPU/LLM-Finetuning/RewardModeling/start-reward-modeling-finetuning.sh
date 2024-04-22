# Configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

python reward_modeling_finetuning.py \
       --model_name_or_path=facebook/opt-350m \
       --output_dir="reward_modeling_ipex_llm" \
       --per_device_train_batch_size=8 \
       --num_train_epochs=1 \
       --gradient_accumulation_steps=16 \
       --gradient_checkpointing=True \
       --learning_rate=1.41e-5 \
       --remove_unused_columns=False \
       --optim="adamw_torch" \
       --logging_steps=10 \
       --evaluation_strategy="steps" \
       --max_length=512
