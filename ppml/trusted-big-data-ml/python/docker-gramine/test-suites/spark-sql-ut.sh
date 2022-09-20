#!/bin/bash
mkdir -p /ppml/trusted-big-data-ml/logs/pyspark/sql

for suite in `cat pyNativeSuccessSuites`
do
    gramine-argv-serializer bash -c "/opt/jdk8/bin/java -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' -Xmx1g org.apache.spark.deploy.SparkSubmit --master 'local[4]' --conf spark.network.timeout=10000000 --conf spark.executor.heartbeatInterval=10000000 --conf spark.python.use.daemon=false --conf spark.python.worker.reuse=false /ppml/trusted-big-data-ml/work/spark-3.1.2/python/pyspark/sql/tests/$suite" > secured_argvs
    gramine-sgx bash 2>&1 | tee logs/pyspark/sql/$suite.log
if [ -n "$(grep "FAILED" logs/pyspark/sql/$suite.log -H -o)" ]; then
    echo "failed"
    exit
done