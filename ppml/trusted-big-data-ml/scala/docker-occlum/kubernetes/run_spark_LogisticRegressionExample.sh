#!/bin/bash

sed -i 's/container-name/spark-LogisticRegression/g' ./executor.yaml
${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://${kubernetes_master_url}:6443 \
    --deploy-mode cluster \
    --name spark-LogisticRegressionExample \
    --class org.apache.spark.examples.ml.LogisticRegressionExample \
    --conf spark.executor.instances=1 \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    --jars local:/opt/spark/examples/jars/scopt_2.12-3.7.1.jar \
    local:/opt/spark/examples/jars/spark-examples_2.12-3.1.2.jar \
    --regParam 0.3 --elasticNetParam 0.8 \
    /opt/spark/data/mllib/sample_libsvm_data.txt

sed -i 's/spark-LogisticRegression/container-name/g' ./executor.yaml

