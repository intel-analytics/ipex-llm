#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-gbt \
    --class com.intel.analytics.bigdl.dllib.example.nnframes.gbt.gbtClassifierTrainingExampleOnCriteoClickLogsDataset \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.5.0-SNAPSHOT \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./driver.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.kubernetes.sgx.log.level=off \
    --conf spark.kubernetes.driverEnv.DRIVER_MEMORY=1g \
    --conf spark.kubernetes.driverEnv.SGX_MEM_SIZE="15GB" \
    --conf spark.kubernetes.driverEnv.SGX_HEAP="1GB" \
    --conf spark.kubernetes.driverEnv.SGX_KERNEL_HEAP="2GB" \
    --conf spark.kubernetes.driverEnv.SGX_THREAD="1024" \
    --conf spark.executorEnv.SGX_MEM_SIZE="12GB" \
    --conf spark.executorEnv.SGX_KERNEL_HEAP="1GB" \
    --conf spark.executorEnv.SGX_HEAP="1GB" \
    --conf spark.executorEnv.SGX_THREAD="1024" \
    --conf spark.task.cpus=6 \
    --conf spark.cores.max=12 \
    --conf spark.executor.instances=2 \
    --conf spark.executorEnv.USING_TMP_HOSTFS=false \
    --executor-cores 6 \
    --executor-memory 1g \
    --conf spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE="5G" \
    --conf spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE="3G" \
    --conf spark.rpc.askTimeout=600s \
    --conf spark.executor.heartbeatInterval=100s \
    local:/opt/spark/examples/jars/spark-examples_2.12-3.1.3.jar \
    -i /host/data/gbt_data -s /host/data/path_to_save -I 20 -d 5
