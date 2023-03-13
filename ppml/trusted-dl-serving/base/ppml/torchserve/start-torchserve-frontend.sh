#!/bin/bash
configFile=""
core=""
SGX_ENABLED="false"
while getopts ":f:c:x" opt
do
    case $opt in
        c)
            configFile=$OPTARG
            ;;
        f)
            core=$OPTARG
            ;;
        x)
            SGX_ENABLED="true"
            ;;
        *)
            echo "Unknown argument passed in: $opt"
            exit 1
            ;;
    esac
done

cd /ppml || exit

if [[ $SGX_ENABLED == "false" ]]; then
    taskset -c "$core" /opt/jdk11/bin/java \
            -Dmodel_server_home=/usr/local/lib/python3.8/dist-packages \
            -cp .:/ppml/torchserve/* \
            -Xmx5g \
            -Xms1g \
            org.pytorch.serve.ModelServer \
            --python /usr/bin/python3 \
            -f "$configFile" \
            -ncs
else
    export sgx_command="/opt/jdk11/bin/java \
            -Dmodel_server_home=/usr/local/lib/python3.8/dist-packages \
            -cp .:/ppml/torchserve/* \
            -Xmx5g \
            -Xms1g \
            org.pytorch.serve.ModelServer \
            --python /usr/bin/python3 \
            -f $configFile \
            -ncs"
    taskset -c "$core" gramine-sgx bash 2>&1 | tee frontend-sgx.log
fi
