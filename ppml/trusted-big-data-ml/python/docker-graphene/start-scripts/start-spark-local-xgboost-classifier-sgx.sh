#!/bin/bash
SGX=1 ./pal_loader bash -c "export RABIT_TRACKER_IP=your_IP_address && /opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/analytics-zoo-0.11.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=2g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.11.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.11.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file /ppml/trusted-big-data-ml/work/analytics-zoo-0.11.0-SNAPSHOT/conf/spark-analytics-zoo.conf \
  --jars /ppml/trusted-big-data-ml/work/analytics-zoo-0.11.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
  --py-files /ppml/trusted-big-data-ml/work/analytics-zoo-0.11.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-python-api.zip \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/examples/pyzoo/xgboost/xgboost_classifier.py \
  -f your_path_of_pima_indians_diabetes_data_csv" | tee test-xgboost-classifier-sgx.log
