#!/bin/bash
cd /ppml

export sgx_command="/opt/jdk8/bin/java \
     -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/examples/jars/*:/ppml/spark-$SPARK_VERSION/jars/*:$BIGDL_HOME/jars/* \
     -Xmx1g org.apache.spark.deploy.SparkSubmit \
     --master local[4] \
     --conf spark.network.timeout=10000000 \
     --conf spark.executor.heartbeatInterval=10000000 \
     --conf spark.python.use.daemon=false \
     --conf spark.python.worker.reuse=false \
     --py-files $BIGDL_HOME/python/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip,$BIGDL_HOME/python/bigdl-dllib-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip,$BIGDL_HOME/python/bigdl-spark_$SPARK_VERSION-$BIGDL_VERSION-python-api.zip \
     /ppml/bigdl-ppml/src/bigdl/ppml/api/simple_query_example.py \
     --app_id 465227134889 \
     --api_key 799072978028 \
     --primary_key_material /ppml/data/SimpleQueryExampleWithSimpleKMS/keys/simple/primaryKey \
     --input_path /ppml/data/SimpleQueryExampleWithSimpleKMS/ppml_test/input \
     --output_path /ppml/data/SimpleQueryExampleWithSimpleKMS/ppml_test/output \
     --input_encrypt_mode AES/CBC/PKCS5Padding \
     --output_encrypt_mode plain_text"
gramine-sgx bash 2>&1 | tee test-python-spark-simplequery.log