#!/bin/bash

set -x

master_host=$SPARK_MASTER_IP
master_port=$SPARK_MASTER_PORT
master_webui_port=$SPARK_MASTER_WEBUI_PORT
secure_passowrd=$SPARK_SECURE_PASSWORD

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
    -Xmx2g \
    org.apache.spark.deploy.master.Master \
    --host $master_host \
    --port $master_port \
    --webui-port $master_webui_port | tee ./spark-master-sgx.log
