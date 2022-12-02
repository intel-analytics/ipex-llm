#!/bin/bash
 
export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/conf/:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/jars/*:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/* -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class org.apache.spark.examples.SparkPi \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --verbose \
    local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 100"
gramine-sgx bash 2>&1 | tee spark-pi-local-sgx.log
