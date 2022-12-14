#!/bin/bash

set -x

if [  "$USING_LOCAL_DATA_KEY" == "true" ]; then
    echo "[INFO] Using local key"
    if [ -z "$LOCAL_DATA_KEY" ]; then
        echo "[ERROR] LOCAL_DATA_KEY is not set!"
        exit 1
    fi
    # Ensure that the file have at least 16 bytes
    filesize=$(stat --printf="%s" $LOCAL_DATA_KEY)
    if [ $? -gt 0 ]; then
        echo "[ERROR] Failed to get the size of $LOCAL_DATA_KEY"
        exit 1
    fi
    if [ $filesize -lt 15 ]; then
        echo "[ERROR] The key should be at least 15 bytes"
        exit 1
    fi
    cmd='data_key=$(cat $LOCAL_DATA_KEY | head -c 15)'
    echo $cmd >> temp_command_file
    echo 'echo $data_key > /dev/attestation/keys/sgx_data_key' >> temp_command_file
else
    echo "[INFO] Using EHSM KMS service to acquire keys"
    # Check if the user set EHSM_URL or not
    if [ -z "$EHSM_URL" ]; then
        echo "[ERROR] Distributed encrypted filesystem is enabled, but EHSM_URL is empty!"
        echo "[INFO] PPML Application Exit!"
        exit 1
    fi

    if [ -z "$APP_ID" ]; then
        echo "[ERROR] Distributed encrypted filesystem is enabled, but APP_ID is empty!"
        echo "[INFO] PPML Application Exit!"
        exit 1
    fi

    if [ -z "$API_KEY" ]; then
        echo "[ERROR] Distributed encrypted filesystem is enabled, but API_KEY is empty!"
        echo "[INFO] PPML Application Exit!"
        exit 1
    fi

    if [ -z "$EHSM_PORT" ]; then
        echo "[INFO] Distributed encrypted filesystem is enabled, but EHSM_PORT is empty!"
        echo "[INFO] Assume EHSM_PORT is 3000"
        echo "export EHSM_PORT=3000" >> temp_command_file
    fi

    # We need to check if encrypted_primary_key and encrypted_data_key exists or not
    if [ -z "$EHSM_PRIMARY_KEY" ]; then
        echo "[INFO] Distributed encrypted filesystem is enabled, but EHSM_PRIMARY_KEY is empty!"
        echo "[INFO] Assume /ppml/encrypted_keys/encrypted_primary_key"
        echo "export EHSM_PRIMARY_KEY=/ppml/encrypted_keys/encrypted_primary_key" >> temp_command_file
    fi

    if [ -z "$EHSM_DATA_KEY" ]; then
        echo "[INFO] Distributed encrypted filesystem is enabled, but EHSM_DATA_KEY is empty!"
        echo "[INFO] Assume /ppml/encrypted_keys/encrypted_data_key"
        echo "export EHSM_DATA_KEY=/ppml/encrypted_keys/encrypted_data_key" >> temp_command_file
    fi

    cmd='data_key=$(python3 /ppml/kms/client.py -api get_data_key_plaintext -ip $EHSM_URL -port $EHSM_PORT -pkp $EHSM_PRIMARY_KEY -dkp $EHSM_DATA_KEY)'
    echo $cmd >> temp_command_file
    echo 'if [ $? -gt 0 ]; then ' >> temp_command_file
    echo '  exit 1' >> temp_command_file
    echo 'fi' >> temp_command_file
    echo 'data_key_u=$(echo $data_key | head -c 15)' >> temp_command_file
    echo 'echo $data_key_u > /dev/attestation/keys/sgx_data_key' >> temp_command_file
fi
