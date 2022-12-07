#!/bin/bash
cd /ppml/trusted-big-data-ml

/opt/jdk8/bin/java \
    -cp "/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/conf/:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/jars/*:/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/*" -Xmx1g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    --class org.apache.spark.examples.SparkPi \
    local:///ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 100
