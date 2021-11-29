#!/bin/bash
status_1_k8s_spark_pi=1
SPARK_LOCAL_IP=192.168.0.112

if [ $status_1_k8s_spark_pi -ne 0 ]; then
  SGX=1 ./pal_loader bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  export SPARK_LOCAL_IP=$SPARK_LOCAL_IP && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-pi-sgx \
    --conf spark.driver.host=$SPARK_LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=10g \
    --conf spark.kubernetes.authenticate.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.executor.instances=2 \
    --executor-cores 8 \
    --total-executor-cores 16 \
    --executor-memory 64G \
    --jars /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar \
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/conf/spark-bigdl.conf \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=16g \
    --class org.apache.spark.examples.SparkPi \
    --verbose \
    local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > k8s-spark-pi-sgx.log
fi
status_1_k8s_spark_pi=$(echo $?)

echo "#### example.1 Excepted result(k8s-spark-pi): Pi is roughly XXX"
echo "---- example.1 Actual result: "
cat k8s-spark-pi-sgx.log | egrep -a "roughly"
