#!/bin/bash
set -x
source /opt/intel/oneapi/setvars.sh
export CCL_WORKER_COUNT=$WORLD_SIZE
source bigdl-llm-init -t
if [ "$WORKER_ROLE" = "launcher" ]
then
  sed "s/:1/ /g" /etc/mpi/hostfile > /home/mpiuser/hostfile
  if [ -d "/model" ];
  then
    MODEL_PARAM="--repo-id-or-model-path ./model"  # otherwise, default to download from HF repo
  fi

  if [ -d "/data/$DATA_SUB_PATH" ];
  then
    DATA_PARAM="/data/$DATA_SUB_PATH" # otherwise, default to download from HF dataset
  fi

  sleep 10
  mpirun \
    -n $WORLD_SIZE \
    -ppn 1 \
    -f /home/mpiuser/hostfile \
    -iface eth0 \
    --bind-to socket \
    -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
    -genv KMP_AFFINITY="granularity=fine,none" \
    -genv KMP_BLOCKTIME=1 \
    -genv TF_ENABLE_ONEDNN_OPTS=1 \
    python qlora_finetuning_cpu.py \
      $MODEL_PARAM $DATA_PARAM > /home/mpiuser/launcher.log 2>&1
  exit_status=$?
  if [ $exit_status -ne 0 ];
  then
    cat /home/mpiuser/launcher.log
    exit $exit_status
  else
    while true
    do
      echo "[INFO] Successfully finished fine-tuning"
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
