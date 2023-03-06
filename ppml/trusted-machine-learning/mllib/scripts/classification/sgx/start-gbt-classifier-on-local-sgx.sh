#!/bin/bash
# source code link: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeClassifierExample.scala

cd /ppml
export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/spark-${SPARK_VERSION}/conf/:/ppml/spark-${SPARK_VERSION}/jars/*:/ppml/spark-${SPARK_VERSION}/examples/jars/* -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class org.apache.spark.examples.ml.GradientBoostedTreeClassifierExample \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --verbose \
    local:///ppml/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000"
gramine-sgx bash 2>&1 | tee gradient-boosted-tree-classfier-local-sgx.log