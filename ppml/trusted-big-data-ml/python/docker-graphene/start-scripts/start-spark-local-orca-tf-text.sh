#!/bin/bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-2.1.0/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.3/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.3/jars/*' \
  -Xmx3g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=3g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-2.1.0/jars/* \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-2.1.0/jars/* \
  --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0/conf/spark-analytics-zoo.conf \
  --jars /ppml/trusted-big-data-ml/work/bigdl-2.1.0/jars/* \
  --py-files /ppml/trusted-big-data-ml/work/bigdl-2.1.0/python/bigdl-orca-spark_3.1.3-2.1.0-python-api.zip \
  --executor-memory 3g \
  --executor-cores 2 \
  --driver-cores 2 \
  /ppml/trusted-big-data-ml/work/examples/pyzoo/orca/learn/tf/basic_text_classification/basic_text_classification.py \
  --cluster_mode local" | tee test-orca-tf-text-sgx.log
