#!/bin/bash
cd /ppml

export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/bigdl-$BIGDL_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --verbose \
    --jars local:///ppml/bigdl-$BIGDL_VERSION/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    local:///ppml/bigdl-$BIGDL_VERSION/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    --inputPath /ppml/data/SimpleQueryExampleWithSimpleKMS/files/people.csv.cbc \
    --outputPath /ppml/data/SimpleQueryExampleWithSimpleKMS/files/output \
    --inputPartitionNum 8 \
    --outputPartitionNum 8 \
    --inputEncryptModeValue AES/CBC/PKCS5Padding \
    --outputEncryptModeValue plain_text \
    --primaryKeyPath /ppml/data/SimpleQueryExampleWithSimpleKMS/files/primaryKey \
    --dataKeyPath /ppml/data/SimpleQueryExampleWithSimpleKMS/files/dataKey \
    --kmsType SimpleKeyManagementService \
    --simpleAPPID 465227134889 \
    --simpleAPIKEY 799072978028"
gramine-sgx bash 2>&1 | tee test-scala-spark-simplequery.log
