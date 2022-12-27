#!/bin/bash
configFile=""
while getopts "c:" opt
do
    case $opt in
        c)
            configFile=$OPTARG
        ;;
    esac
done

cd /ppml
export sgx_command="/opt/jdk11/bin/java \
        -Dmodel_server_home=/usr/local/lib/python3.7/dist-packages \
        -cp .:/ppml/torchserve/* \
        -Xmx30g \
        -Xms30g \
        org.pytorch.serve.ModelServer \
        --python /usr/bin/python3 \
        -f $configFile \
        -ncs"
gramine-sgx bash 2>&1 | tee frontend-sgx.log

