#!/bin/bash
set -x
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

if [ -d "/bigdl/model" ]
then
  MODEL_PARAM="--repo-id-or-model-path /bigdl/model"  # otherwise, default to download from HF repo
fi

if [ -d "/bigdl/data/english_quotes" ];
then
  DATA_PARAM="--dataset /bigdl/data/english_quotes" # otherwise, default to download from HF dataset
fi

if [ "$STANDALONE_DOCKER" = "TRUE" ]
then
  export CONTAINER_IP=$(hostname -i)
  export CPU_CORES=$(nproc)
  source /opt/intel/oneapi/setvars.sh
  export CCL_WORKER_COUNT=$WORKER_COUNT_DOCKER
  export CCL_WORKER_AFFINITY=auto
  export MASTER_ADDR=$CONTAINER_IP
  mpirun \
     -n $CCL_WORKER_COUNT \
     -ppn $CCL_WORKER_COUNT \
     -genv OMP_NUM_THREADS=$((CPU_CORES / CCL_WORKER_COUNT)) \
     -genv KMP_AFFINITY="granularity=fine,none" \
     -genv KMP_BLOCKTIME=1 \
     -genv TF_ENABLE_ONEDNN_OPTS=1 \
     python /bigdl/qlora_finetuning_cpu.py $MODEL_PARAM $DATA_PARAM

else
  source /opt/intel/oneapi/setvars.sh
  export CCL_WORKER_COUNT=$WORLD_SIZE
  export CCL_WORKER_AFFINITY=auto
  if [ "$WORKER_ROLE" = "launcher" ]
  then
    sed "s/:1/ /g" /etc/mpi/hostfile > /home/mpiuser/hostfile
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
      python /bigdl/qlora_finetuning_cpu.py $MODEL_PARAM $DATA_PARAM > /home/mpiuser/launcher.log 2>&1
    exit_status=$?
    if [ $exit_status -ne 0 ];
    then
      cat /home/mpiuser/launcher.log
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
fi


