#!/bin/bash

set -x

master_host=$SPARK_MASTER_IP
master_port=$SPARK_MASTER_PORT
master_webui_port=$SPARK_MASTER_WEBUI_PORT
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`

SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
    -cp "/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*" \
    -Dspark.authenticate=true \
    -Dspark.authenticate.secret=$secure_password \
    -Dspark.network.crypto.enabled=true \
    -Dspark.network.crypto.keyLength=128 \
    -Dspark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    -Dspark.io.encryption.enabled=true \
    -Dspark.io.encryption.keySizeBits=128 \
    -Dspark.io.encryption.keygen.algorithm=HmacSHA1 \
    -Dspark.ssl.enabled=true \
    -Dspark.ssl.port=8043 \
    -Dspark.ssl.keyPassword=$secure_password \
    -Dspark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    -Dspark.ssl.keyStorePassword=$secure_password \
    -Dspark.ssl.keyStoreType=JKS \
    -Dspark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    -Dspark.ssl.trustStorePassword=$secure_password \
    -Dspark.ssl.trustStoreType=JKS \
    -Dspark.worker.timeout=6000 \
    -Xmx2g \
    org.apache.spark.deploy.master.Master \
    --host $master_host \
    --port $master_port \
    --webui-port $master_webui_port" | tee ./spark-master-sgx.log
