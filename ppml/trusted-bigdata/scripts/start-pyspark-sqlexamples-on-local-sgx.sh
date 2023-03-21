#!/bin/bash
status_5_local_spark_basic_sql=0
status_6_local_spark_arrow=1
status_7_local_spark_hive=1
export MALLOC_ARENA_MAX=8
# entry /ppml dir
cd /ppml
export PYSPARK_PYTHON=/usr/bin/python
if [ $status_5_local_spark_basic_sql -ne 0 ]; then
echo "example.5 local spark, Basic SQL"
export sgx_command="/opt/jdk8/bin/java \
  -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* \
  -Xmx1g org.apache.spark.deploy.SparkSubmit \
  --master local[4] \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  /ppml/spark-$SPARK_VERSION/examples/src/main/python/sql/basic.py"
gramine-sgx bash 2>&1 | tee test-sql-basic-sgx.log && \
cat test-sql-basic-sgx.log | egrep '\+\-|Name:' -A10
status_5_local_spark_basic_sql=$(echo $?)
fi

if [ $status_6_local_spark_arrow -ne 0 ]; then
echo "example.6 local spark, Arrow"
export sgx_command="/opt/jdk8/bin/java \
  -cp /ppml/spark-2.1.2/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* \
  -Xmx2g org.apache.spark.deploy.SparkSubmit \
  --master local[4] \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.sql.execution.arrow.enabled=true \
  --conf spark.driver.memory=2g \
  --executor-memory 2g \
  /ppml/spark-$SPARK_VERSION/examples/src/main/python/sql/arrow.py"
gramine-sgx bash 2>&1 | tee test-sql-arrow-sgx.log
status_6_local_spark_arrow=$(echo $?)
fi

if [ $status_7_local_spark_hive -ne 0 ]; then
echo "example.7 local spark, Hive"
export MALLOC_ARENA_MAX=16
export sgx_command="/opt/jdk8/bin/java \
  -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* \
  -Xmx2g org.apache.spark.deploy.SparkSubmit \
  --master local[4] \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.driver.memory=2g \
  --conf spark.sql.broadcastTimeout=30000 \
  --executor-memory 2g \
  /ppml/spark-$SPARK_VERSION/examples/src/main/python/sql/hive.py"
gramine-sgx bash 2>&1 | tee test-sql-hive-sgx.log
status_7_local_spark_hive=$(echo $?)
fi

if [ $status_5_local_spark_basic_sql -ne 0 ]; then
echo "#### example.5 Excepted result(basic.py): 8"
echo "---- example.5 Actual result: "
cat test-sql-basic-sgx.log | egrep -a 'Justin' | wc -l
fi

if [ $status_6_local_spark_arrow -ne 0 ]; then
echo "#### example.6 Excepted result(arrow.py): |    time| id| v1| v2|"
echo "---- example.6 Actual result: "
cat test-sql-arrow-sgx.log | egrep -a '\|\s*time\|'
fi

if [ $status_7_local_spark_hive -ne 0 ]; then
echo "#### example.7 Excepted result(hive.py): |key| value|key| value|"
echo "---- example.7 Actual result: "
cat test-sql-hive-sgx.log | egrep -a '\|key\|.*value\|key\|.*value\|'
fi