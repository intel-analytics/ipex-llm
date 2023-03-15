#!/bin/bash
cd /ppml

export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode cluster \
    --sgx-enabled true \
    --sgx-driver-jvm-memory 1g\
    --sgx-executor-jvm-memory 3g\
    --num-executors 4 \
    --driver-memory 1g \
    --driver-cores 8 \
    --executor-memory 1g \
    --executor-cores 8\
    --conf spark.cores.max=32 \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
    --conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
    --conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --name simple-query-sgx-on-cluster \
    --verbose \
    local://$BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    --inputPath /ppml/data/test_path_do_not_change/simplequery/people.csv \
    --outputPath /ppml/data/test_path_do_not_change/simplequery/output \
    --inputPartitionNum 8 \
    --outputPartitionNum 8 \
    --inputEncryptModeValue plain_text \
    --outputEncryptModeValue plain_text
