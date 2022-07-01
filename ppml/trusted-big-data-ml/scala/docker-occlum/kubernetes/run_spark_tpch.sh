#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-tpch \
    --class main.scala.TpchQuery \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.1.0-SNAPSHOT \
    --conf spark.kubernetes.container.image.pullPolicy="IfNotPresent" \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./driver.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.kubernetes.executor.podNamePrefix="sparktpch" \
    --conf spark.kubernetes.sgx.log.level=off \
    --num-executors 1 \
    --executor-cores 8 \
    --executor-memory 16g \
    --driver-memory 16g \
    --verbose \
    local:/opt/spark/jars/spark-tpc-h-queries_2.12-1.0.jar \
    /host/data/tpch_data/ /host/data/tpch_output/
