#!/bin/bash
cd /ppml
export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
        --sgx-enabled true \
        --deploy-mode client \
        --master $RUNTIME_SPARK_MASTER \
        --sgx-driver-jvm-memory 1g\
        --sgx-executor-jvm-memory 3g\
        --num-executors 2 \
        --driver-memory 1g \
        --driver-cores 8 \
        --executor-memory 1g \
        --executor-cores 8\
        --conf spark.cores.max=64 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.container.image.pullPolicy=Always \
        --class org.apache.spark.examples.sql.SparkSQLExample \
        --name sqlexample-gramine \
        --log-file k8s-spark-sql-example-sgx.log \
        --verbose \
        /ppml/spark-$SPARK_VERSION/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar

echo "#### Excepted result(k8s-spark-sql-example): 10"
echo "---- Actual result: "
cat k8s-spark-sql-example-sgx.log | egrep -a 'Justin' | wc -l