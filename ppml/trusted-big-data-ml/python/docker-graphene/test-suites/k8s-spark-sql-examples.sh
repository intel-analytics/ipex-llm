#!/bin/bash
status_4_k8s_spark_sql_example=1
status_5_k8s_spark_sql_e2e=1

SPARK_LOCAL_IP=$LOCAL_IP
DB_PATH=/ppml/trusted-big-data-ml/work/data/sqlite_example/100w.db

if [ $status_4_k8s_spark_sql_example -ne 0 ]; then
cd /ppml/trusted-big-data-ml
./clean.sh
/graphene/Tools/argv_serializer bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  export SPARK_LOCAL_IP=$SPARK_LOCAL_IP && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.3/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.3/jars/*' \
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
    --jars /ppml/trusted-big-data-ml/work/spark-3.1.3/examples/jars/spark-examples_2.12-3.1.3.jar \
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/conf/spark-bigdl.conf \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=16g \
    --class org.apache.spark.examples.sql.SparkSQLExample \
    /ppml/trusted-big-data-ml/work/spark-3.1.3/examples/jars/spark-examples_2.12-3.1.3.jar" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee k8s-spark-sql-example-sgx.log
fi
status_4_k8s_spark_sql_example=$(echo $?)

if [ $status_5_k8s_spark_sql_e2e -ne 0 ]; then
cd /ppml/trusted-big-data-ml
./clean.sh
/graphene/Tools/argv_serializer bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  export SPARK_LOCAL_IP=$SPARK_LOCAL_IP && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.3/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.3/jars/*:/ppml/trusted-big-data-ml/work/data/sqlite_example/spark-example-sql-e2e.jar' -Xmx10g \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-sql-e2e-sgx \
    --conf spark.driver.host=$SPARK_LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=16g \
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
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-$BIGDL_VERSION/conf/spark-bigdl.conf \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=16g \
    --class test.SqlExample \
    /ppml/trusted-big-data-ml/work/spark-3.1.3/examples/jars/spark-example-sql-e2e.jar \
    $DB_PATH" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee k8s-spark-sql-e2e-100w-sgx.log
fi
status_5_k8s_spark_sql_e2e=$(echo $?)

echo "#### example.4 Excepted result(k8s-spark-sql-example): 10"
echo "---- example.5 Actual result: "
cat k8s-spark-sql-example-sgx.log | egrep -a 'Justin' | wc -l

echo "#### example.4 Excepted result(k8s-spark-sql-e2e): INFO this is result2 count: XXX"
echo "---- example.5 Actual result: "
cat k8s-spark-sql-e2e-100w-sgx.log | egrep -a 'INFO this is result2 count:'
