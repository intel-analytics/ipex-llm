#!/bin/bash
cd /ppml/trusted-big-data-ml
export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/conf/:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/jars/*:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/* \
    -Xmx1g org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/src/main/python/pi.py"
gramine-sgx bash 2>&1 | tee test-pi-sgx.log
cat /ppml/trusted-big-data-ml/test-pi-sgx.log | egrep "roughly"
