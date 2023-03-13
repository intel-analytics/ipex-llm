#!/bin/bash

# Check environment variables
if [ -z "$SPARK_HOME" ]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

if [ -z "$RUNTIME_K8S_SERVICE_ACCOUNT" ]; then
    echo "Please set RUNTIME_K8S_SERVICE_ACCOUNT environment variable"
    exit 1
fi

if [ -z $RUNTIME_K8S_SPARK_IMAGE ]; then
    echo "Please set RUNTIME_K8S_SPARK_IMAGE environment variable"
    exit 1
fi

default_config="--conf spark.kubernetes.authenticate.driver.serviceAccountName=$RUNTIME_K8S_SERVICE_ACCOUNT \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.executor.deleteOnTermination=false \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false"

if [ $secure_password ]; then
   if [ -z $KEYS_PATH ]; then
       echo "Please set $KEYS_PATH environment variable"
       exit 1
   fi
   SSL="--conf spark.authenticate=true \
        --conf spark.authenticate.secret=$secure_password \
        --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
        --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
        --conf spark.authenticate.enableSaslEncryption=true \
        --conf spark.network.crypto.enabled=true \
        --conf spark.network.crypto.keyLength=128 \
        --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
        --conf spark.io.encryption.enabled=true \
        --conf spark.io.encryption.keySizeBits=128 \
        --conf spark.io.encryption.keygen.algorithm=HmacSHA1 \
        --conf spark.ssl.enabled=true \
        --conf spark.ssl.port=8043 \
        --conf spark.ssl.keyPassword=$secure_password \
        --conf spark.ssl.keyStore=$KEYS_PATH/keystore.jks \
        --conf spark.ssl.keyStorePassword=$secure_password \
        --conf spark.ssl.keyStoreType=JKS \
        --conf spark.ssl.trustStore=$KEYS_PATH/keystore.jks \
        --conf spark.ssl.trustStorePassword=$secure_password \
        --conf spark.ssl.trustStoreType=JKS"
else
   SSL=""
fi

set -x

spark_submit_command="${SPARK_HOME}/bin/spark-submit \
        $default_config \
        $SSL \
        $*"

echo "spark_submit_command $spark_submit_command"
bash -c "$spark_submit_command"
