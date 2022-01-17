#!/bin/bash

SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -Xms8g -Xmx8g -cp $SPARK_HOME/jars/*:$BIGDL_HOME/jars/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.example.VflLogisticRegression -d data/diabetes-vfl-1.csv"
