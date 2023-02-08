#!/bin/bash
cd /ppml

export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --sgx-enabled true \
    --sgx-driver-jvm-memory 20g\
    --sgx-executor-jvm-memory 10g\
    --num-executors 4 \
    --driver-memory 10g \
    --driver-cores 8 \
    --executor-memory 10g \
    --executor-cores 8\
    --conf spark.cores.max=32 \
    --conf spark.kubernetes.executor.container.image=10.239.45.10/arda/intelanalytics/bigdl-ppml-trusted-bigdata-gramine-reference-32g:$BIGDL_VERSION \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
    --conf spark.executor.extraClassPath=/ppml/jars/* \
    --conf spark.driver.extraClassPath=/ppml/jars/* \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --name simple-query-sgx-on-client \
    --verbose \
    local:///ppml/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    --inputPath /ppml/data/test_path_do_not_change/simplequery/people.csv \
    --outputPath /ppml/data/test_path_do_not_change/simplequery/output \
    --inputPartitionNum 8 \
    --outputPartitionNum 8 \
    --inputEncryptModeValue plain_text \
    --outputEncryptModeValue plain_text
