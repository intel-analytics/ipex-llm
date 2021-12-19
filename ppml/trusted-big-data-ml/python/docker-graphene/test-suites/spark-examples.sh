#!/bin/bash
cd /ppml/trusted-big-data-ml
status_1_scala_spark_pi=1

if [ $status_1_scala_spark_pi -ne 0 ]; then
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --class org.apache.spark.examples.SparkPi \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-pi-sgx.log
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
