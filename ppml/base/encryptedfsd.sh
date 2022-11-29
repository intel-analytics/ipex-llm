#!/bin/bash

set -x

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
    export EHSM_PORT=3000
fi

# We need to check if encrypted_primary_key and encrypted_data_key exists or not
if [ -z "$EHSM_PRIMARY_KEY" ]; then
    echo "[INFO] Distributed encrypted filesystem is enabled, but EHSM_PRIMARY_KEY is empty!"
    echo "[INFO] Assume /ppml/encrypted_keys/encrypted_primary_key"
    export EHSM_PRIMARY_KEY=/ppml/encrypted_keys/encrypted_primary_key
fi

if [ -z "$EHSM_DATA_KEY" ]; then
    echo "[INFO] Distributed encrypted filesystem is enabled, but EHSM_DATA_KEY is empty!"
    echo "[INFO] Assume /ppml/encrypted_keys/encrypted_data_key"
    export EHSM_DATA_KEY=/ppml/encrypted_keys/encrypted_data_key
fi


cmd='data_key=$(python3 /ppml/kms/client.py -api get_data_key_plaintext -ip $EHSM_URL -port $EHSM_PORT -pkp $EHSM_PRIMARY_KEY -dkp $EHSM_DATA_KEY)'
echo $cmd > temp_command_file
echo 'if [ $? -gt 0 ]; then ' >> temp_command_file
echo '  exit 1' >> temp_command_file
echo 'fi' >> temp_command_file
echo 'data_key_u=$(echo $data_key | head -c 15)' >> temp_command_file
echo 'echo $data_key_u > /dev/attestation/keys/default' >> temp_command_file
