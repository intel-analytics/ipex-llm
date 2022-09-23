#!/bin/bash
 
/opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.3/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.3/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.3/examples/jars/*' -Xmx16g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[2] \
    --executor-memory 8g \
    --driver-memory 8g \
    --class org.apache.spark.examples.SparkPi \
    --verbose \
    local:///ppml/trusted-big-data-ml/work/spark-3.1.3/examples/jars/spark-examples_2.12-3.1.3.jar 100