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
cd /ppml/trusted-big-data-ml
export sgx_command="/opt/jdk8/bin/java \
        -cp /ppml/trusted-big-data-ml/work/bigdl-${BIGDL_VERSION}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/jars/commons-lang3-3.10.jar:/ppml/trusted-big-data-ml/work/bigdl-${BIGDL_VERSION}/jars/*:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/conf/:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/jars/*:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/* \
        -Xmx10g org.apache.spark.deploy.SparkSubmit \
        --master local[4] \
        /ppml/trusted-big-data-ml/fl/start-fgboost-server.py --port $port --client_num $client_num"
gramine-sgx bash 2>&1 | tee fgboost-server-sgx.log
