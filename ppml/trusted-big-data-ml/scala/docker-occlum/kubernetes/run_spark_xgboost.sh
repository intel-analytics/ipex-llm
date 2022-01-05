#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-xgboost \
    --class com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierTrainingExample \
    --conf spark.executor.instances=1 \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.task.cpus=2 \
    --executor-cores 2 \
    --executor-memory 2g \
    --total-executor-cores 2 \
    --driver-cores 1 \
    --driver-memory 2g \
    local:/bin/jars/bigdl-dllib-spark_3.1.2-0.14.0-SNAPSHOT.jar \
    /host/data/iris.data 2  100 /host/data/xgboost_model_to_be_saved
