#!/bin/bash
 
/opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/conf/:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/jars/*:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/*' -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[2] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class org.apache.spark.examples.SparkPi \
    --verbose \
    local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-3.1.2.jar 100