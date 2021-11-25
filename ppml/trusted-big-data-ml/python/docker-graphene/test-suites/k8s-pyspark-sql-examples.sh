#!/bin/bash
status_2_k8s_pyspark_sql_basic=1
status_3_k8s_pyspark_sql_e2e=1

SPARK_LOCAL_IP=192.168.0.112
DB_PATH=/ppml/trusted-big-data-ml/work/data/sqlite_example/test_100w.db

if [ $status_2_k8s_pyspark_sql_basic -ne 0 ]; then
SGX=1 ./pal_loader bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  export SPARK_LOCAL_IP=$SPARK_LOCAL_IP && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name pyspark-sql-basic-sgx \
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
    --verbose \
    /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/basic.py" 2>&1 > k8s-pyspark-sql-basic-sgx.log
fi
status_2_k8s_pyspark_sql_basic=$(echo $?)

if [ $status_3_k8s_pyspark_sql_e2e -ne 0 ]; then
SGX=1 ./pal_loader bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  export SPARK_LOCAL_IP=$SPARK_LOCAL_IP && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' -Xmx10g \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name pyspark-sql-e2e-sgx \
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
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/conf/spark-bigdl.conf \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=16g \
    --verbose \
    local:///ppml/trusted-big-data-ml/work/examples/customer_profile.py \
    --db_path $DB_PATH" 2>&1 > k8s-pyspark-sql-e2e-100w-sgx.log
fi
status_3_k8s_pyspark_sql_e2e=$(echo $?)

echo "#### example.2 Excepted result(k8s-pyspark-sql-basic): 8"
echo "---- example.2 Actual result: "
cat k8s-pyspark-sql-basic-sgx.log | egrep -a 'Justin' | wc -l

echo "#### example.3 Excepted result(k8s-pyspark-sql-e2e): INFO this is results count: XXX"
echo "---- example.3 Actual result: "
cat k8s-pyspark-sql-e2e-100w-sgx.log | egrep -a 'INFO this is results count:'
