#!/bin/bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
  -Xmx8g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=8g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
  --conf spark.rpc.message.maxSize=190 \
  --conf spark.network.timeout=10000000 \
  --conf spark.executor.heartbeatInterval=10000000 \
  --py-files /ppml/trusted-big-data-ml/work/bigd-python-api.zip,/ppml/trusted-big-data-ml/work/examples/bigdl/lenet/lenet.py \
  --jars /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
  --driver-cores 2 \
  --total-executor-cores 2 \
  --executor-cores 2 \
  --executor-memory 8g \
  /ppml/trusted-big-data-ml/work/examples/bigdl/lenet/lenet.py \
  --dataPath /ppml/trusted-big-data-ml/work/data/mnist \
  --maxEpoch 2" | tee test-bigdl-lenet-sgx.log
