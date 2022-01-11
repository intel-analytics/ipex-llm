#!/bin/bash
SGX=1 graphene-sgx bash -c "/opt/jdk8/bin/java -cp /opt/bigdl-0.14.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
    com.intel.analytics.bigdl.ppml.example.VflLogisticRegression -d /opt/data/diabetes-vfl-1.csv"