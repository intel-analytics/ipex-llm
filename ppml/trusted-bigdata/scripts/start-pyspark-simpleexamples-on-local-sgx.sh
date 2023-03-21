#!/bin/bash
status_1_local_spark_pi=1
status_2_local_spark_wordcount=0
export MALLOC_ARENA_MAX=12
cd /ppml

if [ $status_1_local_spark_pi -ne 0 ]; then
echo "sexample.1 local spark, pi"
export sgx_command="/opt/jdk8/bin/java \
   -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* \
   -Xmx1g org.apache.spark.deploy.SparkSubmit \
   --master local[4] \
   --conf spark.python.use.daemon=false \
   --conf spark.python.worker.reuse=false \
   /ppml/spark-$SPARK_VERSION/examples/src/main/python/pi.py"
gramine-sgx bash 2>&1 | tee test-pi-sgx.log
cat test-pi-sgx.log | egrep 'roughly'
status_1_local_spark_pi=$(echo $?)
fi


if [ $status_2_local_spark_wordcount -ne 0 ]; then
echo "example.2 local spark, test-wordcount"
export PYSPARK_PYTHON=/usr/bin/python
export sgx_command="/opt/jdk8/bin/java \
   -cp /ppml/spark-$SPARK_VERSION/conf/:/ppml/spark-$SPARK_VERSION/jars/*:/ppml/spark-$SPARK_VERSION/examples/jars/* \
   -Xmx1g org.apache.spark.deploy.SparkSubmit \
   --master local[4] \
   --conf spark.python.use.daemon=false \
   --conf spark.python.worker.reuse=false \
   /ppml/spark-$SPARK_VERSION/examples/src/main/python/wordcount.py \
   /ppml/examples/helloworld.py"
gramine-sgx bash 2>&1 | tee test-wordcount-sgx.log
cat test-wordcount-sgx.log | egrep 'import'
status_2_local_spark_wordcount=$(echo $?)
fi

if [ $status_1_local_spark_pi -ne 0 ]; then
echo "#### sexample.1 Excepted result(Pi): Pi is roughly XXX"
echo "---- sexample.1 Actual result: "
cat test-pi-sgx.log | egrep 'roughly'
fi

if [ $status_2_local_spark_wordcount -ne 0 ]; then
echo -e "#### example.2 Excepted result(WordCount): import: XXX"
echo "---- example.2 Actual result: "
cat test-wordcount-sgx.log | egrep -a 'import.*: [0-9]*$'
fi