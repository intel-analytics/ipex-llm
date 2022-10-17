#!/bin/bash
port=8981
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
        -cp '/ppml/work/bigdl-2.2.0-SNAPSHOT/jars/grpc-protobuf-1.47.0.jar:/ppml/work/bigdl-2.2.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.2.0-SNAPSHOT.jar:/ppml/work/spark-3.1.2/jars/commons-lang3-3.10.jar:/ppml/work/bigdl-2.2.0-SNAPSHOT/jars/*:/ppml/work/spark-3.1.2/conf/:/ppml/work/spark-3.1.2/jars/*'\
        -Xmx10g org.apache.spark.deploy.SparkSubmit\
        --verbose \
        --master 'local[4]'\
        /ppml/fl/start-fgboost-server.py --port $port --client_num $client_num"
gramine-sgx bash 2>&1 | tee fgboost-server-sgx.log
