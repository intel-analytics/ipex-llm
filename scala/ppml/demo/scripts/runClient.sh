#!/bin/bash
clientType=$1
clientId=$2

case "$clientType" in
    hfl)
        case "$clientId" in
            1)
                export script="com.intel.analytics.bigdl.ppml.fl.example.HFLLogisticRegression -d /ppml/trusted-big-data-ml/work/data/diabetes-hfl-1.csv"
                ;;
            2)
                export script="com.intel.analytics.bigdl.ppml.fl.example.HFLLogisticRegression -d /ppml/trusted-big-data-ml/work/data/diabetes-hfl-2.csv"
                ;;
        esac
        ;;
    vfl)
        case "$clientId" in
            1)
                export script="com.intel.analytics.bigdl.ppml.fl.example.VFLLogisticRegression -d /ppml/trusted-big-data-ml/work/data/diabetes-vfl-1.csv -c 1"
                ;;
            2)
                export script="com.intel.analytics.bigdl.ppml.fl.example.VFLLogisticRegression -d /ppml/trusted-big-data-ml/work/data/diabetes-vfl-2.csv -c 2"
                ;;
        esac
        ;;
esac

/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java -Xms2g -Xmx2g -XX:ActiveProcessorCount=4 -cp \
  $SPARK_HOME/jars/*:$BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-jar-with-dependencies.jar \
  $script" > /ppml/trusted-big-data-ml/secured-argvs
bash /ppml/trusted-big-data-ml/init.sh
SGX=1 /ppml/trusted-big-data-ml/pal_loader bash
