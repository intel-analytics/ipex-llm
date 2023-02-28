#!/bin/bash
port=9000
core=""
SGX_ENABLED="false"
while getopts ":c:p:x" opt
do
    case $opt in
        p)
            port=$OPTARG
            ;;
        c)
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
    taskset -c "$core" /usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ts/model_service_worker.py --sock-type unix --sock-name /tmp/.ts.sock."${port}" --metrics-config /usr/local/lib/python3.8/dist-packages/ts/configs/metrics.yaml
else
    export sgx_command="/usr/bin/python3 /usr/local/lib/python3.8/dist-packages/ts/model_service_worker.py --sock-type tcp --port $port --metrics-config /ppml/metrics.yaml"
    taskset -c "$core" gramine-sgx bash 2>&1 | tee backend-sgx.log
fi
