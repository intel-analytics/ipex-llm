#!/bin/bash
# this is to run alpaca qlora on k8s
set -x
source /opt/intel/oneapi/setvars.sh
export CCL_WORKER_COUNT=$WORLD_SIZE
source bigdl-llm-init -t
if [ "$WORKER_ROLE" = "launcher" ]
then
  sed "s/:1/ /g" /etc/mpi/hostfile > /home/mpiuser/hostfile
  cd /bigdl/alpaca-qlora
  sleep 10
  export MASTER_ADDR=$(hostname -i)
  export CPU_CORES=$(nproc)
  source /opt/intel/oneapi/setvars.sh
  mpirun \
    -n $WORLD_SIZE \
    -ppn 1 \
    -f /home/mpiuser/hostfile \
    -iface eth0 \
    --bind-to socket \
    -genv OMP_NUM_THREADS=$((CPU_CORES / WORLD_SIZE))\
    -genv KMP_AFFINITY="granularity=fine,none" \
    -genv KMP_BLOCKTIME=1 \
    -genv TF_ENABLE_ONEDNN_OPTS=1 \
    python alpaca_qlora_finetuning_cpu.py \
      --base_model '/bigdl/model'  \
      --data_path "/bigdl/data" \
      --output_dir "/home/mpiuser/finetuned_model" \
      --batch_size 128 \
      --micro_batch_size 16 > /home/mpiuser/launcher.log 2>&1
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
