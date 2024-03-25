#/bin/bash
source ipex-llm-init
unset OMP_NUM_THREADS # deepspeed will set it for each instance automatically
source /opt/intel/oneccl/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export WORLD_SIZE=2 # run 1 instance per SPR socket, thus 2 instances on 2 sockets, 96 cores
export MASTER_ADDR=127.0.0.1
export CCL_ZE_IPC_EXCHANGE=sockets
export DS_ACCELERATOR="cpu"
export CCL_WORKER_AFFINITY=auto
unset KMP_AFFINITY # deepspeed will set it for each instance automatically
export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_PROCESS_LAUNCHER=none

deepspeed \
  --bind_cores_to_rank \
  --bind_core_list 0-95 \
  deepspeed_autotp.py
