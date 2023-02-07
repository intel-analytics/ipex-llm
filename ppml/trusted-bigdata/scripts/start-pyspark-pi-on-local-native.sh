#!/bin/bash
cd /ppml

/opt/jdk8/bin/java \
    -cp "/ppml/spark-${SPARK_VERSION}/conf/:/ppml/spark-${SPARK_VERSION}/jars/*:/ppml/spark-${SPARK_VERSION}/examples/jars/*" \
    -Xmx1g org.apache.spark.deploy.SparkSubmit \
    --master local[4] \
    /ppml/spark-${SPARK_VERSION}/examples/src/main/python/pi.py

