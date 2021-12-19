#!/bin/bash
status_5_local_spark_basic_sql=1
status_6_local_spark_arrow=1
status_7_local_spark_hive=1

# entry /ppml/trusted-big-data-ml dir
cd /ppml/trusted-big-data-ml

if [ $status_5_local_spark_basic_sql -ne 0 ]; then
echo "example.5 local spark, Basic SQL"
SGX=1 ./pal_loader bash -c "export PYSPARK_PYTHON=/usr/bin/python && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/basic.py" 2>&1 > test-sql-basic-sgx.log && \
  cat test-sql-basic-sgx.log | egrep '\+\-|Name:' -A10
status_5_local_spark_basic_sql=$(echo $?)
fi

if [ $status_6_local_spark_arrow -ne 0 ]; then
echo "example.6 local spark, Arrow"
SGX=1 ./pal_loader bash -c "export PYSPARK_PYTHON=/usr/bin/python && \
  export ARROW_PRE_0_15_IPC_FORMAT=0 && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.sql.execution.arrow.enabled=true \
  --conf spark.driver.memory=2g \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/arrow.py" 2>&1 > test-sql-arrow-sgx.log
status_6_local_spark_arrow=$(echo $?)
fi

if [ $status_7_local_spark_hive -ne 0 ]; then
echo "example.7 local spark, Hive"
SGX=1 ./pal_loader bash -c "export PYSPARK_PYTHON=/usr/bin/python && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.driver.memory=2g \
  --conf spark.sql.broadcastTimeout=30000 \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/hive.py" 2>&1 > test-sql-hive-sgx.log
status_7_local_spark_hive=$(echo $?)
fi


echo "#### example.5 Excepted result(basic.py): 8"
echo "---- example.5 Actual result: "
cat test-sql-basic-sgx.log | egrep -a 'Justin' | wc -l

echo "#### example.6 Excepted result(arrow.py): |    time| id| v1| v2|"
echo "---- example.6 Actual result: "
cat test-sql-arrow-sgx.log | egrep -a '\|\s*time\|'

echo "#### example.7 Excepted result(hive.py): |key| value|key| value|"
echo "---- example.7 Actual result: "
cat test-sql-hive-sgx.log | egrep -a '\|key\|.*value\|key\|.*value\|'
