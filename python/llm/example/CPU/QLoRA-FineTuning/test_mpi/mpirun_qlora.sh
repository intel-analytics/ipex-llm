#export CCL_WORKER_COUNT=8
#export CCL_WORKER_AFFINITY=0,1,2,3,4,5,6,7
export MASTER_ADDR=10.112.242.183
export MASTER_PORT=12345
# export RANK=$1
# export WORLD_SIZE=$2


mpirun -hostfile ./hosts -n 2 -ppn 1 \
 --bind-to socket \
 -genv OMP_NUM_THREADS=48 \
 -genv KMP_AFFINITY="granularity=fine,none" \
 -genv KMP_BLOCKTIME=1 \
 python qlora_finetuning_cpu-mpi.py \
 --batch_size 16 \
 --micro_batch_size 8 \
 --steps 100 \
 --base_model '/home/llm/models/Llama-2-7b-chat-hf'  \
 --data_path '/root/wangjian/llm/finetune/data/alpaca_data/' \
 --output_dir '/root/wangjian/llm/finetune/output-mpi'
