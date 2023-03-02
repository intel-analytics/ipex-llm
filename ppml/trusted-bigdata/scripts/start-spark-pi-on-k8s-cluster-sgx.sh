#!/bin/bash
cd /ppml

export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 6g\
        --sgx-executor-jvm-memory 6g\
        --num-executors 2 \
        --driver-memory 4g \
        --driver-cores 8 \
        --executor-memory 4g \
        --executor-cores 8\
        --conf spark.cores.max=16 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --conf spark.kubernetes.file.upload.path=file:///tmp \
        --verbose \
        --log-file spark-pi-cluster-sgx.log \
        --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
        local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000