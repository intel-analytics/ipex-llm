#!/bin/bash
sleep 30

set -x
source /opt/intel/oneapi/setvars.sh
export CCL_WORKER_COUNT=$WORLD_SIZE
export CCL_WORKER_AFFINITY=auto
export SAVE_PATH="/ppml/output"

mpirun \
    -n $WORLD_SIZE \
    -ppn 1 \
    -f /home/mpiuser/hostfile \
    -iface eth0 \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -genv KMP_AFFINITY="granularity=fine,none" \
    -genv KMP_BLOCKTIME=1 \
    -genv TF_ENABLE_ONEDNN_OPTS=1 \
    sudo -E python /ppml/worker_quote_generate.py --user_report_data ppml > $SAVE_PATH/quote.log 2>&1