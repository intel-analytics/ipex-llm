#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name pyspark-tpch \
    --conf spark.executor.instances=2 \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.5.0-SNAPSHOT \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./driver.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.sgx.log.level=off \
    --executor-memory 1g \
    --driver-memory 1g \
    --num-executors 2 \
    --executor-cores 4 \
    --conf spark.cores.max=8 \
    --conf spark.sql.shuffle.partitions=16 \
    --conf spark.executorEnv.USING_TMP_HOSTFS=true \
    --conf spark.kubernetes.driverEnv.USING_TMP_HOSTFS=true \
    --conf spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE="1G" \
    --conf spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE="10G" \
    --py-files local:/py-examples/tpch/tpch.zip \
    local:/py-examples/tpch/main.py \
    /host/data/tpch_data/ /host/data/tpch_output/ true
