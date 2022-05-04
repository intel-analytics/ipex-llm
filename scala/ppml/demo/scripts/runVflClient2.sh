#!/bin/bash

/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java -Xms2g -Xmx2g -cp $SPARK_HOME/jars/*:$BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.fl.example.VFLLogisticRegression -d /ppml/trusted-big-data-ml/work/data/diabetes-vfl-2.csv --hasLabel false" > /ppml/trusted-big-data-ml/secured-argvs
bash /ppml/trusted-big-data-ml/init.sh
SGX=1 /ppml/trusted-big-data-ml/pal_loader bash
