#!/bin/bash
port=8980
client_num=2

while getopts "p:c:" opt
do
    case $opt in
        p)
            port=$OPTARG
        ;;
        c)
            client_num=$OPTARG
        ;;
    esac
done
cd /ppml
export sgx_command="/opt/jdk8/bin/java\
        -cp '/ppml/work/spark-3.1.2/conf/:/ppml/work/spark-3.1.2/jars/*'\
        -Xmx10g org.apache.spark.deploy.SparkSubmit\
        --master 'local[4]'\
        /ppml/fl/start-fl-server.py -p $port -c $client_num"
gramine-sgx bash 2>&1 | tee fl-server-sgx.log
