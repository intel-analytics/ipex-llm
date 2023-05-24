#!/bin/bash
configFile=""
core=""
while getopts ":f:c:" opt
do
    case $opt in
        c)
            configFile=$OPTARG
            ;;
        f)
            core=$OPTARG
            ;;
    esac
done

cd /ppml
export sgx_command="/opt/jdk11/bin/java \
        -Dmodel_server_home=/usr/local/lib/python3.8/dist-packages \
        -cp .:/ppml/torchserve/* \
        -Xmx5g \
        -Xms1g \
        org.pytorch.serve.ModelServer \
        --python /usr/bin/python3 \
        -f $configFile \
        -ncs"
taskset -c $core gramine-sgx bash 2>&1 | tee frontend-sgx.log

