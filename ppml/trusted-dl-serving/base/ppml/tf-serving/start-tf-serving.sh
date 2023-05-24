#!/bin/bash
model_path=""
model_name=""
rest_port=""
tensorflow_inter_op_parallelism=""
tensorflow_intra_op_parallelism=""
parameters=""
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION

usage() {
    echo "Usage: $0 [-t <model path>] [-n <model name>] [-e <tensorflow inter op parallelism>] [-a <tensorflow intra op parallelism>] [-r <rest api port>] [-p <parameters for tf-serving>] [-X <enable sgx>] [-A <enable attestation>]"
}

while getopts ":t:n:e:a:r:p:XA" opt
do
    case $opt in
        t)
            model_path=$OPTARG
            ;;
        n)
            model_name=$OPTARG
            ;;
        e)
            tensorflow_inter_op_parallelism=$OPTARG
            ;;
        a)
            tensorflow_intra_op_parallelism=$OPTARG
            ;;
        r)
            rest_port=$OPTARG
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
if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        rm /ppml/temp_command_file || true
        bash attestation.sh
        bash temp_command_file
    fi
    /usr/bin/tensorflow_model_server --model_base_path=${model_path} --model_name=${model_name} --rest_api_port=${rest_port} --tensorflow_inter_op_parallelism=${tensorflow_inter_op_parallelism} --tensorflow_intra_op_parallelism=${tensorflow_intra_op_parallelism} ${parameters}
    rm /ppml/temp_command_file || true
else
    cd /ppml
    export sgx_command="/usr/bin/tensorflow_model_server --model_base_path=${model_path} --model_name=${model_name} --rest_api_port=${rest_port} --tensorflow_inter_op_parallelism=${tensorflow_inter_op_parallelism} --tensorflow_intra_op_parallelism=${tensorflow_intra_op_parallelism} ${parameters}"
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
    gramine-sgx bash 2>&1 | tee tfserving-sgx.log
    rm /ppml/temp_command_file || true
fi

