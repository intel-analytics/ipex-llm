#!/bin/bash

port=8980
client_num=2

while getopts "p:c:" opt
do
    case $opt in
        p)
            port=$OPTARG
        ;;
        c)
            client_num=$OPTARG
        ;;
    esac
done
cd /ppml/trusted-big-data-ml
bash clean.sh
/graphene/Tools/argv_serializer bash -c " /opt/jdk8/bin/java\
        -cp '/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*'\
        -Xmx10g org.apache.spark.deploy.SparkSubmit\
        --master 'local[4]'\
        /ppml/trusted-big-data-ml/fl/start-fgboost-server.py --port $port --client_num $client_num" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee fgboost-server-sgx.log

