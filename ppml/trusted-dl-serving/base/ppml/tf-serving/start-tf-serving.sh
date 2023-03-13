#!/bin/bash
model_path=""
model_name=""
rest_port=""
tensorflow_inter_op_parallelism=""
tensorflow_intra_op_parallelism=""
parameters=""
SGX_ENABLED="false"

usage() {
    echo "Usage: $0 [-t <model path>] [-n <model name>] [-e <tensorflow inter op parallelism>] [-a <tensorflow intra op parallelism>] [-r <rest api port>] [-p <parameters for tf-serving>] [-x <enable sgx>]"
}

while getopts ":t:n:e:a:r:p:x" opt
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
        x)
            SGX_ENABLED="true"
            ;;
        *)
            echo "Error: unknown positional arguments"
            usage
            ;;
    esac
done
if [[ $SGX_ENABLED == "false" ]]; then
    /usr/bin/tensorflow_model_server --model_base_path=${model_path} --model_name=${model_name} --rest_api_port=${rest_port} --tensorflow_inter_op_parallelism=${tensorflow_inter_op_parallelism} --tensorflow_intra_op_parallelism=${tensorflow_intra_op_parallelism} ${parameters}
else
    cd /ppml
    ./init.sh
    export sgx_command="/usr/bin/tensorflow_model_server --model_base_path=${model_path} --model_name=${model_name} --rest_api_port=${rest_port} --tensorflow_inter_op_parallelism=${tensorflow_inter_op_parallelism} --tensorflow_intra_op_parallelism=${tensorflow_intra_op_parallelism} ${parameters}"
    gramine-sgx bash 2>&1 | tee tfserving-sgx.log
fi

