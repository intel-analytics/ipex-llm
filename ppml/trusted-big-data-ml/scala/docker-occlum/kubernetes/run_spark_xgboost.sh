#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-xgboost \
    --class com.intel.analytics.bigdl.dllib.example.nnframes.xgboost.xgbClassifierTrainingExample \
    --conf spark.executor.instances=1 \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.2.0-SNAPSHOT \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./driver.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.kubernetes.sgx.log.level=off \
    --conf spark.task.cpus=2 \
    --executor-cores 6 \
    --executor-memory 3g \
    --driver-memory 2g \
    --conf spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE="2G" \
    --conf spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE="3G" \
    local:/bin/jars/bigdl-dllib-spark_3.1.3-2.2.0-SNAPSHOT.jar \
    /host/data/xgboost_data 2  100 /host/data/xgboost_model_to_be_saved
