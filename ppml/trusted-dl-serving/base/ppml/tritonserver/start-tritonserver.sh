#!/bin/bash
model=""
parameters=""
SGX_ENABLED="false"

usage() {
    echo "Usage: $0 [-m <model>] [-p <parameters for tritonserver>] [-x <enable sgx>]"
}

while getopts ":m:p:x" opt
do
    case $opt in
        m)
            model=$OPTARG
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
echo $parameters
if [[ $SGX_ENABLED == "false" ]]; then
    /opt/tritonserver/bin/tritonserver --model-repository=${model} ${parameters}
else
    cd /ppml
    ./init.sh
    export sgx_command="/opt/tritonserver/bin/tritonserver --model-repository=${model} ${parameters}"
    gramine-sgx bash 2>&1 | tee tritonserver-sgx.log
fi

