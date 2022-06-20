#!/bin/bash
command="$@"
default_config="--conf spark.driver.host=$LOCAL_IP \
        --conf spark.driver.port=$RUNTIME_DRIVER_PORT \
        --conf spark.driver.cores=$RUNTIME_DRIVER_CORES \
        --conf spark.driver.memory=$RUNTIME_DRIVER_MEMORY \
        --conf spark.executor.cores=$RUNTIME_EXECUTOR_CORES \
        --conf spark.executor.memory=$RUNTIME_EXECUTOR_MEMORY \
        --conf spark.executor.instances=$RUNTIME_EXECUTOR_INSTANCES \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false"
k8s_config="--conf spark.kubernetes.sgx.enabled=$SGX_ENABLED \
        --conf spark.kubernetes.sgx.driver.mem=$SGX_DRIVER_MEM \
        --conf spark.kubernetes.sgx.driver.jvm.mem=$SGX_DRIVER_JVM_MEM \
        --conf spark.kubernetes.sgx.executor.mem=$SGX_EXECUTOR_MEM \
        --conf spark.kubernetes.sgx.executor.jvm.mem=$SGX_EXECUTOR_JVM_MEM \
        --conf spark.kubernetes.sgx.log.level=$SGX_LOG_LEVEL \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
        --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/spark-driver-template.yaml \
        --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
        --conf spark.kubernetes.executor.deleteOnTermination=false"
if [ $secure_password ]; then
   SSL="--conf spark.authenticate=true \
    --conf spark.authenticate.secret=$secure_password \
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret"  \
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret"  \
    --conf spark.authenticate.enableSaslEncryption=true \
    --conf spark.network.crypto.enabled=true  \
    --conf spark.network.crypto.keyLength=128  \
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    --conf spark.io.encryption.enabled=true \
    --conf spark.io.encryption.keySizeBits=128 \
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1 \
    --conf spark.ssl.enabled=true \
    --conf spark.ssl.port=8043 \
    --conf spark.ssl.keyPassword=$secure_password \
    --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks  \
    --conf spark.ssl.keyStorePassword=$secure_password \
    --conf spark.ssl.keyStoreType=JKS \
    --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.trustStorePassword=$secure_password \
    --conf spark.ssl.trustStoreType=JKS"
else
   SSL=""
fi
spark_submit_command="${JAVA_HOME}/bin/java \
        -cp ${SPARK_HOME}/conf/:${SPARK_HOME}/jars/* \
        -Xmx${RUNTIME_DRIVER_MEMORY} \
        org.apache.spark.deploy.SparkSubmit \
        $SSL \
        $default_config"
if [ "$SPARK_MODE" == "cluster" ] || [ "$SPARK_MODE" == "client" ]; then
    spark_submit_command="${spark_submit_command} ${k8s_config}"
fi
spark_submit_command="${spark_submit_command} ${command}"

if [ "$SGX_ENABLED" == "true" ] && [ "$SPARK_MODE" != "cluster" ]; then
    ./clean.sh
    /graphene/Tools/argv_serializer bash -c "$spark_submit_command" > /ppml/trusted-big-data-ml/secured-argvs
 
    ./init.sh
    SGX=1 ./pal_loader bash 2>&1 | tee bigdl-ppml-submit.log
else
    $spark_submit_command 2>&1 | tee bigdl-ppml-submit.log
fi
