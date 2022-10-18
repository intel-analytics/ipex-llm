#!/bin/bash
cd /ppml/trusted-big-data-ml

export sgx_command="/opt/jdk8/bin/java \
     -cp '/ppml/trusted-big-data-ml/work/data/SimpleQueryWithSimpleKMS/files/bigdl-ppml-spark_3.1.2-2.2.0-SNAPSHOT.jar:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/bigdl-2.2.0-SNAPSHOT/jars/*' \
     -Xmx1g org.apache.spark.deploy.SparkSubmit \
     --master local[4] \
     --conf spark.network.timeout=10000000 \
     --conf spark.executor.heartbeatInterval=10000000 \
     --conf spark.python.use.daemon=false \
     --conf spark.python.worker.reuse=false \
     --jars local://$SPARK_HOME/examples/jars/scopt_2.12-3.7.1.jar,$(echo $BIGDL_HOME/jars/* |tr ' ' ',' | sed "s#${BIGDL_HOME}#local://${BIGDL_HOME}#g") \
     --py-files /ppml/trusted-big-data-ml/work/bigdl-2.2.0-SNAPSHOT/python/bigdl-ppml-spark_3.1.2-2.2.0-SNAPSHOT-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-2.2.0-SNAPSHOT/python/bigdl-spark_3.1.2-2.2.0-SNAPSHOT-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-2.2.0-SNAPSHOT/python/bigdl-dllib-spark_3.1.2-2.2.0-SNAPSHOT-python-api.zip \
     /ppml/trusted-big-data-ml/work/data/zhoujian/code/simple_query_example.py --simple_app_id 465227134889 --simple_app_key 799072978028 --primary_key_path /ppml/trusted-big-data-ml/work/data/zhoujian/keys/simple/primaryKey --data_key_path /ppml/trusted-big-data-ml/work/data/zhoujian/keys/simple/dataKey --input_path /ppml/trusted-big-data-ml/work/data/zhoujian/ppml_test/input --output_path /ppml/trusted-big-data-ml/work/data/zhoujian/ppml_test/output --input_encrypt_mode AES/CBC/PKCS5Padding --output_encrypt_mode plain_text"

gramine-sgx bash 2>&1 | tee test-python-spark-simplequery.log
