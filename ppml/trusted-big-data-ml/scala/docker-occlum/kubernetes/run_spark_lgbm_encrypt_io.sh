#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-lgbm-encrypt-io \
    --class com.intel.analytics.bigdl.dllib.example.nnframes.lightGBM.LgbmClassifierTrain \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:2.5.0-SNAPSHOT \
    --conf spark.kubernetes.container.image.pullPolicy="IfNotPresent" \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./driver.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.kubernetes.executor.podNamePrefix="spark-lgbm-encrypt-io" \
    --conf spark.kubernetes.sgx.log.level=off \
    --num-executors 2 \
    --executor-cores 4 \
    --conf spark.cores.max=8 \
    --executor-memory 1g \
    --conf spark.kubernetes.driverEnv.SGX_DRIVER_JVM_MEM_SIZE="1G" \
    --conf spark.executorEnv.SGX_EXECUTOR_JVM_MEM_SIZE="5G" \
    --py-files local:/py-examples/bigdl.zip \
    local:/py-examples/encrypted_lightgbm_model_io.py \
    --app_id 123456654321 \
    --api_key 123456654321 \
    --primary_key_material /host/data/key/simple_encrypted_primary_key \
    --input_path /host/data/iris.data \
    --output_path /host/data/model \
    --input_encrypt_mode plain_text \
    --output_encrypt_mode plain_text \
    --kms_type SimpleKeyManagementService


