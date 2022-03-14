#!/bin/bash

SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -Xms2g -Xmx2g -cp $SPARK_HOME/jars/*:$BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-jar-with-dependencies.jar \
 com.intel.analytics.bigdl.ppml.FLServer"
