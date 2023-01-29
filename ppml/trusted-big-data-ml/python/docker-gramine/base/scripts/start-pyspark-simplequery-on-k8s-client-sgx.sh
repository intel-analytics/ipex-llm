#!/bin/bash
cd /ppml/trusted-big-data-ml

secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
export TF_MKL_ALLOC_MAX_BYTES=10737418240
export SPARK_LOCAL_IP=$LOCAL_IP

export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/conf/:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/jars/*:/ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/* \
    -Xmx1g org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name pyspark-simple-query \
    --conf spark.driver.host=$LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=32g \
    --conf spark.executor.cores=8 \
    --conf spark.executor.memory=32g \
    --conf spark.executor.instances=2 \
    --conf spark.cores.max=32 \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/spark-driver-template.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.executor.jvm.mem=12g \
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
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --jars local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar,local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-dllib-spark_$SPARK_VERSION-$BIGDL_VERSION.jar,local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-friesian-spark_$SPARK_VERSION-$BIGDL_VERSION.jar,local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-grpc-spark_$SPARK_VERSION-$BIGDL_VERSION.jar,local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-orca-spark_$SPARK_VERSION-$BIGDL_VERSION.jar,local:///ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/bigdl-serving-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
    --py-files /ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/python/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/python/bigdl-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/python/bigdl-dllib-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip \
    /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/code/simple_query_example.py \
    --simple_app_id 465227134889 \
    --simple_app_key 799072978028 \
    --primary_key_path /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/keys/simple/primaryKey \
    --data_key_path /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/keys/simple/dataKey \
    --input_path /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/ppml_test/input \
    --output_path /ppml/trusted-big-data-ml/work/data/SimpleQueryExampleWithSimpleKMS/ppml_test/output \
    --input_encrypt_mode AES/CBC/PKCS5Padding \
    --output_encrypt_mode plain_text"
gramine-sgx bash 2>&1 | tee test-python-k8s-spark-simplequery.log
