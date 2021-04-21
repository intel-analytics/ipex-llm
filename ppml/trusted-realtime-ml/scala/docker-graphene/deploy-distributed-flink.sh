#!/bin/bash
set -x

JOB_MANAGER_MAX_LOOP_TIME=210
TASK_MANAGER_MAX_LOOP_TIME=450

source environment.sh

echo "### phase.1 distribute the keys and password"
echo ">>> $MASTER"
ssh root@$MASTER "rm -rf $KEYS_PATH && rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && mkdir -p $AZ_PPML_PATH"
scp -r $SOURCE_ENCLAVE_KEY_PATH root@$MASTER:$ENCLAVE_KEY_PATH
scp -r $SOURCE_KEYS_PATH root@$MASTER:$KEYS_PATH
scp -r $SOURCE_SECURE_PASSWORD_PATH root@$MASTER:$SECURE_PASSWORD_PATH
for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "rm -rf $KEYS_PATH && rm -rf $KEYS_PATH && rm -rf $SECURE_PASSWORD_PATH && mkdir -p $AZ_PPML_PATH"
    scp -r $SOURCE_ENCLAVE_KEY_PATH root@$worker:$ENCLAVE_KEY_PATH
    scp -r $SOURCE_KEYS_PATH root@$worker:$KEYS_PATH
    scp -r $SOURCE_SECURE_PASSWORD_PATH root@$worker:$SECURE_PASSWORD_PATH
  done
echo "### phase.1 distribute the keys and password finished successfully"

echo "### phase.2 deploy the flink components"

echo ">>> $MASTER, start flink-jobmanager"
ssh root@$MASTER "docker run \
      --privileged \
      --net=host \
      --cpuset-cpus="3-5" \
      --oom-kill-disable \
      --device=/dev/gsgx \
      --device=/dev/sgx/enclave \
      --device=/dev/sgx/provision \
      -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
      -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
      -v $KEYS_PATH:/ppml/trusted-realtime-ml/java/work/keys \
      -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/java/work/password \
      --name=flink-job-manager \
      -e SGX_MEM_SIZE=32G \
      -e FLINK_JOB_MANAGER_IP=$MASTER \
      -e FLINK_JOB_MANAGER_REST_PORT=8081 \
      -e FLINK_JOB_MANAGER_RPC_PORT=6123 \
      -e CORE_NUM=3 \
      $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /ppml/trusted-realtime-ml/java && ./init-java.sh && ./start-flink-jobmanager.sh && tail -f /dev/null'"


JOB_MANAGER_ELAPSED_TIME=0
while ! ssh root@$MASTER "nc -z $MASTER 8081"; do
    { set +x; } 2>/dev/null
    if [ $JOB_MANAGER_ELAPSED_TIME -gt $JOB_MANAGER_MAX_LOOP_TIME ] ; then
        echo "Error: Flink job manager port 8081 is unavailable."
        break
    fi
    JOB_MANAGER_ELAPSED_TIME=$((JOB_MANAGER_ELAPSED_TIME+10))
    set -x
    sleep 10
done
{ set +x; } 2>/dev/null
if [ $JOB_MANAGER_ELAPSED_TIME -le $JOB_MANAGER_MAX_LOOP_TIME ] ; then
    echo ">>> $MASTER, flink-jobmanager started successfully."
fi
set -x

for worker in ${WORKERS[@]}
  do
    echo ">>> $worker"
    ssh root@$worker "docker run -itd \
        --privileged \
        --net=host \
        --cpuset-cpus="6-30" \
        --oom-kill-disable \
        --device=/dev/gsgx \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
        -v $KEYS_PATH:/ppml/trusted-realtime-ml/redis/work/keys \
        -v $KEYS_PATH:/ppml/trusted-realtime-ml/java/work/keys \
        -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/redis/work/password \
        -v $SECURE_PASSWORD_PATH:/ppml/trusted-realtime-ml/java/work/password \
        --name=flink-task-manager-$worker \
        -e SGX_MEM_SIZE=64G \
        -e FLINK_JOB_MANAGER_IP=$MASTER \
        -e FLINK_JOB_MANAGER_REST_PORT=8081 \
        -e FLINK_JOB_MANAGER_RPC_PORT=6123 \
        -e FLINK_TASK_MANAGER_IP=$worker \
        -e FLINK_TASK_MANAGER_DATA_PORT=6124 \
        -e FLINK_TASK_MANAGER_RPC_PORT=6125 \
        -e FLINK_TASK_MANAGER_TASKSLOTS_NUM=1 \
        -e CORE_NUM=25 \
        $TRUSTED_CLUSTER_SERVING_DOCKER bash -c 'cd /ppml/trusted-realtime-ml/java && ./init-java.sh && ./start-flink-taskmanager.sh'"
  done

for worker in ${WORKERS[@]}
    do
        TASK_MANAGER_ELAPSED_TIME=0
        while ! ssh root@$MASTER "nc -z $MASTER 6124"; do
            { set +x; } 2>/dev/null
            if [ $TASK_MANAGER_ELAPSED_TIME -gt $TASK_MANAGER_MAX_LOOP_TIME ] ; then
                echo "Error: Flink TASK manager port 6124 is unavailable."
                break
            fi
            TASK_MANAGER_ELAPSED_TIME=$((TASK_MANAGER_ELAPSED_TIME+10))
            set -x
            sleep 10
        done
        { set +x; } 2>/dev/null
        if [ $TASK_MANAGER_ELAPSED_TIME -le $TASK_MANAGER_MAX_LOOP_TIME ] ; then
            echo ">>> $worker, flink-task-manager-$worker started successfully."
        fi
        set -x
    done
