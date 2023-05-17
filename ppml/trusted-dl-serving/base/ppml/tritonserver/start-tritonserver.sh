#!/bin/bash
model=""
parameters=""
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION

usage() {
    echo "Usage: $0 [-m <model>] [-p <parameters for tritonserver>] [-X <enable sgx>] [-A <enable attestation>]"
}

while getopts ":m:p:XA" opt
do
    case $opt in
        m)
            model=$OPTARG
            ;;
        p)
            parameters=$OPTARG
            ;;
        X)
            SGX_ENABLED="true"
            ;;
        A)
            ATTESTATION="true"
            ;;
        *)
            echo "Error: unknown positional arguments"
            usage
            ;;
    esac
done
echo $parameters
if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        rm /ppml/temp_command_file || true
        bash attestation.sh
        bash temp_command_file
    fi
    /opt/tritonserver/bin/tritonserver --model-repository=${model} ${parameters}
else
    cd /ppml
    export sgx_command="/opt/tritonserver/bin/tritonserver --model-repository=${model} ${parameters}"
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
    ./init.sh
    gramine-sgx bash 2>&1 | tee tritonserver-sgx.log
    rm /ppml/temp_command_file || true
    
fi

