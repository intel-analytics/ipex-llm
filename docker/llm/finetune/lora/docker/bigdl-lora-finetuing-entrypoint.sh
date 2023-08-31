#!/bin/bash
set -x
source /opt/intel/oneapi/setvars.sh
export CCL_WORKER_COUNT=$WORLD_SIZE
export CCL_WORKER_AFFINITY=auto

if [ "$WORKER_ROLE" = "launcher" ]
then
  sed "s/:1/ /g" /etc/mpi/hostfile > /home/mpiuser/hostfile
  export DATA_PATH="/ppml/data/$DATA_SUB_PATH"
  export SAVE_PATH="/ppml/output"
  sleep 10
  mpirun \
    -n $WORLD_SIZE \
    -ppn 1 \
    -f /home/mpiuser/hostfile \
    -iface eth0 \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -genv KMP_AFFINITY="granularity=fine,none" \
    -genv KMP_BLOCKTIME=1 \
    -genv TF_ENABLE_ONEDNN_OPTS=1 \
    python /ppml/lora_finetune.py \
      --base_model '/ppml/model/'  \
      --data_path "$DATA_PATH" \
      --output_dir "$SAVE_PATH/finetuned_model" \
      --micro_batch_size $MICRO_BATCH_SIZE \
      --bf16 > $SAVE_PATH/launcher.log 2>&1
  exit_status=$?
  if [ $exit_status -ne 0 ];
  then
    cat $SAVE_PATH/launcher.log
    exit $exit_status
  else
    while true
    do
      echo "[INFO] Successfully finished training"
      sleep 900
    done
  fi
elif [ "$WORKER_ROLE" = "trainer" ]
then
  export LOCAL_RANK=$(cut -d "-" -f6 <<< "$LOCAL_POD_NAME")
  export PMI_SIZE=$WORLD_SIZE
  export PMI_RANK=$LOCAL_RANK
  /usr/sbin/sshd -De -f /home/mpiuser/.sshd_config
fi

