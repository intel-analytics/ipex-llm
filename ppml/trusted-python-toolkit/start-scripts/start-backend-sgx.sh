#!/bin/bash
port=9000
while getopts "p:" opt
do
    case $opt in
        p)
            port=$OPTARG
        ;;
    esac
done

cd /ppml
export sgx_command="/usr/bin/python3 /usr/local/lib/python3.7/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port --metrics-config /ppml/metrics.yaml"
gramine-sgx bash 2>&1 | tee backend-sgx.log

