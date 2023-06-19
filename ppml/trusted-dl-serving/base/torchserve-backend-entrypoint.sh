#!/bin/bash
port=$BACKEND_PORT
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION
FRONTEND_IP=$FRONTEND_IP
FRONTEND_MANAGEMENT_PORT=$FRONTEND_MANAGEMENT_PORT
MODEL_NAME=$MODEL_NAME
local_pod_ip=$( hostname -I | awk '{print $1}' )

cd /ppml || exit
# Set PCCS conf
if [ "$PCCS_URL" != "" ] ; then
    echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
    echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
fi

if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        rm /ppml/temp_command_file || true
        bash attestation.sh
        bash temp_command_file
    fi
    /usr/bin/python3 /usr/local/lib/python3.9/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port --host $local_pod_ip --metrics-config /ppml/metrics.yaml --frontend-ip $FRONTEND_IP --frontend-management-port $FRONTEND_MANAGEMENT_PORT --model-name $MODEL_NAME
else
    export sgx_command="/usr/bin/python3 /usr/local/lib/python3.9/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port --host $local_pod_ip --metrics-config /ppml/metrics.yaml --frontend-ip $FRONTEND_IP --frontend-management-port $FRONTEND_MANAGEMENT_PORT --model-name $MODEL_NAME"
    if [ "$ATTESTATION" = "true" ]; then
          # Also consider ENCRYPTEDFSD condition
          rm /ppml/temp_command_file || true
          bash attestation.sh
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
          fi
          echo $sgx_command >>temp_command_file
          export sgx_command="bash temp_command_file"
    else
          # ATTESTATION is false
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            # ATTESTATION false, encrypted-fsd true
            rm /ppml/temp_command_file || true
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
            echo $sgx_command >>temp_command_file
            export sgx_command="bash temp_command_file"
          fi
    fi
    gramine-sgx bash 2>&1 | tee backend-sgx.log
    rm /ppml/temp_command_file || true
fi

