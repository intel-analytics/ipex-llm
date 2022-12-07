#!/bin/bash
cd /ppml/trusted-big-data-ml

export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/conf/:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/jars/*:/ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/*:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/* -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --verbose \
    --jars local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    --inputPath /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/files/people.csv.cbc \
    --outputPath /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/files/output \
    --inputPartitionNum 8 \
    --outputPartitionNum 8 \
    --inputEncryptModeValue AES/CBC/PKCS5Padding \
    --outputEncryptModeValue plain_text \
    --primaryKeyPath /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/files/primaryKey \
    --dataKeyPath /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/files/dataKey \
    --kmsType SimpleKeyManagementService \
    --simpleAPPID 465227134889 \
    --simpleAPIKEY 799072978028"
gramine-sgx bash 2>&1 | tee test-scala-spark-simplequery.log
