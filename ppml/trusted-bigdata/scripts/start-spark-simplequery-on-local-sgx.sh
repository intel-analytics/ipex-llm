#!/bin/bash
cd /ppml

export sgx_command="/opt/jdk8/bin/java \
        -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* -Xmx16g \
        org.apache.spark.deploy.SparkSubmit \
        --master local[4] \
        --executor-memory 8g \
        --driver-memory 8g \
        --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --verbose \
        local:///ppml/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
        --inputPath /ppml/data/test_path_do_not_change/simplequery/people.csv \
        --outputPath /ppml/data/test_path_do_not_change/simplequery/output \
        --inputPartitionNum 8 \
        --outputPartitionNum 8 \
        --inputEncryptModeValue plain_text \
        --outputEncryptModeValue plain_text"
gramine-sgx bash 2>&1 | tee test-scala-spark-simplequery.log