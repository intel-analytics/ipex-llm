#!/bin/bash
port=9000
core=""
while getopts ":c:p:" opt
do
    case $opt in
        p)
            port=$OPTARG
            ;;
        c)
            core=$OPTARG
            ;;
    esac
done

cd /ppml
export sgx_command="/usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port --metrics-config /ppml/metrics.yaml"
taskset -c $core gramine-sgx bash 2>&1 | tee backend-sgx.log

