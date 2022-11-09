#!/bin/bash
status_2_k8s_spark_sql_example=1

SPARK_LOCAL_IP=$LOCAL_IP
if [ $status_2_k8s_spark_sql_example -ne 0 ]; then
cd /ppml/trusted-big-data-ml
export TF_MKL_ALLOC_MAX_BYTES=10737418240
export SPARK_LOCAL_IP=$SPARK_LOCAL_IP
export sgx_command="/opt/jdk8/bin/java \
    -cp /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/conf/:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/jars/*:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/* \
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
    --jars /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar \
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/conf/spark-bigdl.conf \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=16g \
    --class org.apache.spark.examples.sql.SparkSQLExample \
    /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar"
gramine-sgx bash 2>&1 | tee k8s-spark-sql-example-sgx.log
fi
status_2_k8s_spark_sql_example=$(echo $?)

echo "#### example.2 Excepted result(k8s-spark-sql-example): 10"
echo "---- example.2 Actual result: "
cat k8s-spark-sql-example-sgx.log | egrep -a 'Justin' | wc -l

