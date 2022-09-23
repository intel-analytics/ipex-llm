#!/bin/bash
 
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-memory 64g \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-memory 64g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.3/examples/jars/spark-examples_2.12-3.1.3.jar 3000