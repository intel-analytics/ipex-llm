#!/bin/bash
cd /ppml/trusted-big-data-ml
status_1_scala_spark_pi=1

if [ $status_1_scala_spark_pi -ne 0 ]; then
export sgx_command="/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/conf/:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/jars/*:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master local[4] \
  --class org.apache.spark.examples.SparkPi \
  /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar"
gramine-sgx bash 2>&1 | tee test-scala-spark-pi-sgx.log
fi
status_1_scala_spark_pi=$(echo $?)

# echo Results
echo "###########################"
echo "###########################"
echo "########  Results   #######"
echo "###########################"
echo "###########################"

echo "example.1 status_1_scala_spark_pi"
echo -e "Excepted Result: Pi is roughly XXX \n Actual Result:"
cat test-scala-spark-pi-sgx.log | egrep -a 'roughly'
