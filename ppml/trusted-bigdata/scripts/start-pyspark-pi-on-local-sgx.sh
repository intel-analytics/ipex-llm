#!/bin/bash
cd /ppml
export MALLOC_ARENA_MAX=12
export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/spark-${SPARK_VERSION}/conf/:/ppml/spark-${SPARK_VERSION}/jars/*:/ppml/spark-${SPARK_VERSION}/examples/jars/* \
    -Xmx1g org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    /ppml/spark-${SPARK_VERSION}/examples/src/main/python/pi.py"
gramine-sgx bash 2>&1 | tee test-pi-sgx.log
cat /ppml/test-pi-sgx.log | egrep "roughly"
