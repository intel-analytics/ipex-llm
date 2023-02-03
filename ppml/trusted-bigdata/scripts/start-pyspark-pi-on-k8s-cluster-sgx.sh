#!/bin/bash
cd /ppml

secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin` && \
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 20g\
        --sgx-executor-jvm-memory 20g\
        --driver-memory 20g \
        --driver-cores 8 \
        --executor-memory 20g \
        --num-executors 1 \
        --executor-cores 8\
        --conf spark.cores.max=8 \
        --conf spark.pyspark.python=python \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.file.upload.path=file:///tmp \
        --verbose \
        local:///ppml/spark-${SPARK_VERSION}/examples/src/main/python/pi.py
