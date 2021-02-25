#!/bin/bash

set -x

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
        -Xmx10g \
        -Dbigdl.mklNumThreads=1 \
        -XX:ActiveProcessorCount=24 \
        org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
        --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class com.intel.analytics.bigdl.models.lenet.Train \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
        -f /ppml/trusted-big-data-ml/work/data \
        -b 64 \
        -e 1 | tee spark.local.sgx.log
