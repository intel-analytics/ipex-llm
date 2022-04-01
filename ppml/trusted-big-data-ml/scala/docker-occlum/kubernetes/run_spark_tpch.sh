#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-tpch-k8s-test \
    --class main.scala.TpchQuery \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.1.0-SNAPSHOT \
	--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.podNamePrefix="sparktpch" \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.driverEnv.SGX_MEM_SIZE="20GB" \
    --conf spark.executorEnv.SGX_MEM_SIZE="20GB" \
    --num-executors 2 \
    --executor-cores 8 \
    --executor-memory 16g \
    --driver-memory 16g \

