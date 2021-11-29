#!/bin/bash
cd /ppml/trusted-big-data-ml

SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.driver.memory=8g \
  --conf spark.rpc.message.maxSize=190 \
  --conf spark.network.timeout=10000000 \
  --conf spark.executor.heartbeatInterval=10000000 \
  --properties-file /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local://${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/examples/dllib/lenet/lenet.py \
  --driver-cores 2 \
  --total-executor-cores 2 \
  --executor-cores 2 \
  --executor-memory 8g \
  /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/examples/dllib/lenet/lenet.py \
  --dataPath /ppml/trusted-big-data-ml/work/data/mnist \
  --maxEpoch 2" 2>&1 | tee test-bigdl-lenet-sgx.log && \
  cat test-bigdl-lenet-sgx.log | egrep -a "Accuracy"

