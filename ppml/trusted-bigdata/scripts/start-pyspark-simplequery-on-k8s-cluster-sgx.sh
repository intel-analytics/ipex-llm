#!/bin/bash
cd /ppml

export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode cluster \
    --sgx-enabled true \
    --sgx-driver-jvm-memory 1g\
    --sgx-executor-jvm-memory 3g\
    --num-executors 4 \
    --driver-memory 1g \
    --driver-cores 8 \
    --executor-memory 1g \
    --executor-cores 8\
    --conf spark.cores.max=32 \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --py-files $BIGDL_HOME/python/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip,$BIGDL_HOME/python/bigdl-dllib-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip,$BIGDL_HOME/python/bigdl-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --name pyspark-simple-query-sgx-on-cluster \
    --verbose \
    local:///ppml/bigdl-ppml/src/bigdl/ppml/api/simple_query_example.py \
    --app_id 465227134889 \
    --api_key 799072978028 \
    --primary_key_material /ppml/data/SimpleQueryExampleWithSimpleKMS/keys/simple/primaryKey \
    --input_path /ppml/data/SimpleQueryExampleWithSimpleKMS/ppml_test/input \
    --output_path /ppml/data/SimpleQueryExampleWithSimpleKMS/ppml_test/output \
    --input_encrypt_mode AES/CBC/PKCS5Padding \
    --output_encrypt_mode plain_text