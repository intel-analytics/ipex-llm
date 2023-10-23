export MASTER_ADDR=127.0.0.1
export CCL_ZE_IPC_EXCHANGE=sockets
export OMP_NUM_THREADS=28
torchrun --standalone \
         --nnodes=1 \
         --nproc-per-node 4 \
         test_deepspeed.py
