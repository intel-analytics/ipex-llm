#!/bin/bash
port=9000
core=""
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION
while getopts ":c:p:XA" opt
do
    case $opt in
        p)
            port=$OPTARG
            ;;
        c)
            core=$OPTARG
            ;;
        X)
            SGX_ENABLED="true"
            ;;
        A)
            ATTESTATION="true"
            ;;
        *)
            echo "Unknown argument passed in: $opt"
            exit 1
            ;;
    esac
done

cd /ppml || exit

if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        rm /ppml/temp_command_file || true
        bash attestation.sh
        bash temp_command_file
    fi
    taskset -c "$core" /usr/bin/python3 /usr/local/lib/python3.9/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port  --metrics-config /usr/local/lib/python3.9/dist-packages/ts/configs/metrics.yaml
else
    export sgx_command="/usr/bin/python3 /usr/local/lib/python3.9/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port --metrics-config /ppml/metrics.yaml"
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
    taskset -c "$core" gramine-sgx bash 2>&1 | tee backend-sgx.log
    rm /ppml/temp_command_file || true
fi
