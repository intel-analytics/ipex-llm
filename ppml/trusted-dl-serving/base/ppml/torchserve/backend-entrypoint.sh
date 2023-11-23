#!/bin/bash
port=$BACKEND_PORT
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION
FRONTEND_IP=$FRONTEND_IP
FRONTEND_PORT=$FRONTEND_PORT
MODEL_NAME=$MODEL_NAME
SAVED_ON_DISK=$SAVED_ON_DISK
SECURED_DIR=$SECURED_DIR
local_pod_ip=$( hostname -I | awk '{print $1}' )

cd /ppml || exit
./init.sh

# Set PCCS conf
if [ "$PCCS_URL" != "" ] ; then
    echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v4/' > /etc/sgx_default_qcnl.conf
    echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
fi

MODEL_FILE="/tmp/model/torchserve/${MODEL_NAME}.mar"
DECRYPTION_KEY="/tmp/key/torchserve/${AES_KEY}"

cmd="/usr/bin/python3 /usr/local/lib/python3.9/dist-packages/ts/model_service_worker.py --sock-type tcp --port ${port} --host ${local_pod_ip} --metrics-config /ppml/metrics.yaml --frontend-ip ${FRONTEND_IP} --frontend-port ${FRONTEND_PORT} --model-name ${MODEL_NAME} --model-file ${MODEL_FILE}"

if [[ $MODEL_DECRYPTION == "true" ]]; then
    cmd+=" --model-decryption --decryption-key ${DECRYPTION_KEY}"
fi

if [[ $SAVED_ON_DISK == "true" ]]; then
    cmd+=" -s --secured_dir ${SECURED_DIR}"
fi

echo $cmd

if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        rm /ppml/temp_command_file || true
        bash attestation.sh
        bash temp_command_file
    fi
    eval $cmd
else
    export sgx_command=$cmd
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

