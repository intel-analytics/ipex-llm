#!/bin/bash

SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -Xms8g -Xmx8g -cp /ppml/trusted-big-data-ml/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 com.intel.analytics.bigdl.ppml.example.HflLogisticRegression -d data/diabetes-hfl-2.csv"
