#!/bin/bash

set -x

spark_master=$SPARK_MASTER
driver_port=$SPARK_DRIVER_PORT
block_manager_port=$SPARK_BLOCK_MANAGER_PORT
driver_host=$SPARK_DRIVER_IP
driver_block_manager_port=$SPARK_DRIVER_BLOCK_MANAGER_PORT
secure_passowrd=$SPARK_SECURE_PASSWORD

export SPARK_HOME=/ppml/trusted-big-data-ml/work/spark-2.4.3

SGX=1 ./pal_loader /opt/jdk8/bin/java \
    -cp "/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*" \
    -Dspark.authenticate=true \
    -Dspark.authenticate.secret=$secure_passowrd \
    -Dspark.network.crypto.enabled=true \
    -Dspark.network.crypto.keyLength=128 \
    -Dspark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    -Dspark.io.encryption.enabled=true \
    -Dspark.io.encryption.keySizeBits=128 \
    -Dspark.io.encryption.keygen.algorithm=HmacSHA1 \
    -Dspark.ssl.enabled=true \
    -Dspark.ssl.port=8043 \
    -Dspark.ssl.keyPassword=$secure_passowrd \
    -Dspark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    -Dspark.ssl.keyStorePassword=$secure_passowrd \
    -Dspark.ssl.keyStoreType=JKS \
    -Dspark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    -Dspark.ssl.trustStorePassword=$secure_passowrd \
    -Dspark.ssl.trustStoreType=JKS \
    -Xmx10g \
    org.apache.spark.deploy.SparkSubmit \
    --master $spark_master \
    --conf spark.driver.port=$driver_port \
    --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
    --conf spark.worker.timeout=600 \
    --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
    --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
    --conf spark.starvation.timeout=250000 \
    --conf spark.blockManager.port=$block_manager_port \
    --conf spark.driver.host=$driver_host \
    --conf spark.driver.blockManager.port=$driver_block_manager_port \
    --conf spark.network.timeout=1900s \
    --conf spark.executor.heartbeatInterval=1800s \
    --class com.intel.analytics.bigdl.models.lenet.Train \
    --executor-cores 4 \
    --total-executor-cores 4 \
    --executor-memory 12G \
    /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
    -f /ppml/trusted-big-data-ml/work/data \
    -b 64 -e 1 | tee ./spark-driver-sgx.log
