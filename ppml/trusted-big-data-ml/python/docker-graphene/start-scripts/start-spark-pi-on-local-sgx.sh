#!/bin/bash
 
export SGX_ENABLED=true
unset SPARK_MODE
bash bigdl-ppml-submit.sh \
        --master local[2] \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
