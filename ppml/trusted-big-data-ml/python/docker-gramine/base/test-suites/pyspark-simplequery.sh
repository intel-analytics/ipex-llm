#!/bin/bash
cd /ppml/trusted-big-data-ml

export sgx_command="/opt/jdk8/bin/java \
     -cp /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/conf/:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/*:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/jars/*:/ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/jars/* \
     -Xmx1g org.apache.spark.deploy.SparkSubmit \
     --master local[4] \
     --conf spark.network.timeout=10000000 \
     --conf spark.executor.heartbeatInterval=10000000 \
     --conf spark.python.use.daemon=false \
     --conf spark.python.worker.reuse=false \
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
gramine-sgx bash 2>&1 | tee test-python-spark-simplequery.log
