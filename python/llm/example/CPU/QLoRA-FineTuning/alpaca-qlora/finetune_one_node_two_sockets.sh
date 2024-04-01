export MASTER_ADDR=127.0.0.1
export SOCKET_CORES=48

source ipex-llm-init -t
mpirun -n 2 \
 --bind-to socket \
 -genv OMP_NUM_THREADS=$SOCKET_CORES \
 -genv KMP_AFFINITY="granularity=fine,none" \
 -genv KMP_BLOCKTIME=1 \
 python alpaca_qlora_finetuning_cpu.py \
 --gradient_checkpointing False \
 --batch_size 128 \
 --micro_batch_size 8 \
 --max_steps -1 \
 --base_model "meta-llama/Llama-2-7b-hf" \
 --data_path "yahma/alpaca-cleaned" \
 --output_dir "./ipex-llm-qlora-alpaca"

