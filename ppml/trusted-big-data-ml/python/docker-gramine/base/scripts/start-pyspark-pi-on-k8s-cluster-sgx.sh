#!/bin/bash
cd /ppml/trusted-big-data-ml

deploy_mode=cluster && \
driver_name=pyspark-pi-on-k8s-cluster-sgx && \
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin` && \
export SPARK_LOCAL_IP=$LOCAL_IP && \
/opt/jdk8/bin/java \
    -cp $SPARK_HOME/conf/:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
    -Xmx10g org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode $deploy_mode \
    --name $driver_name \
    --conf spark.driver.host=$LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=32g \
    --conf spark.executor.cores=8 \
    --conf spark.executor.memory=32g \
    --conf spark.executor.instances=4 \
    --conf spark.cores.max=32 \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/spark-driver-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.executor.jvm.mem=6g \
    --conf spark.kubernetes.sgx.driver.jvm.mem=4g \
    --conf spark.authenticate=true \
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
    --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.keyStorePassword=$secure_password \
    --conf spark.ssl.keyStoreType=JKS \
    --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.trustStorePassword=$secure_password \
    --conf spark.ssl.trustStoreType=JKS \
    local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/src/main/python/pi.py 2>&1 | tee pyspark-pi-on-k8s-cluster-sgx.log
