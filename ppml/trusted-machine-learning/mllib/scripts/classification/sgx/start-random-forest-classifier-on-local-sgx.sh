#!/bin/bash
# source code link: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/RandomForestClassifierExample.scala

cd /ppml

if [ ! -d "data/mllib" ]
then
    mkdir -p data/mllib
fi

input_file="data/mllib/sample_libsvm_data.txt"

if [ -f "$input_file" ]; then
    echo "Input file exists."
else
    echo "Input file not exists, downloading the file"
    wget https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_libsvm_data.txt -O "$input_file"
fi

export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/spark-${SPARK_VERSION}/conf/:/ppml/spark-${SPARK_VERSION}/jars/*:/ppml/spark-${SPARK_VERSION}/examples/jars/* -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class org.apache.spark.examples.ml.RandomForestClassifierExample \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --verbose \
    local:///ppml/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000"
gramine-sgx bash 2>&1 | tee random-forest-classfier-local-sgx.log
