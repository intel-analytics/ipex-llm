#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-tpch \
    --class com.intel.analytics.bigdl.ppml.examples.tpch.TpchQuery \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.5.0-SNAPSHOT \
    --conf spark.kubernetes.container.image.pullPolicy="IfNotPresent" \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./driver.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.kubernetes.executor.podNamePrefix="sparktpch" \
    --conf spark.kubernetes.sgx.log.level=off \
    --num-executors 2 \
    --executor-cores 4 \
    --conf spark.cores.max=8 \
    --conf spark.sql.shuffle.partitions=16 \
    --executor-memory 4g \
    --driver-memory 1g \
    --conf spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE="1G" \
    --conf spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE="4G" \
    --verbose \
    local:/opt/spark/jars/spark-tpc-h-queries_2.12-1.0.jar \
    /host/data/tpch_data/ /host/data/tpch_output/ plain_text plain_text
