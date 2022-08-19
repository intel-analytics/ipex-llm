#!/bin/bash
cd /ppml/trusted-big-data-ml
/graphene/Tools/argv_serializer bash -c " /opt/jdk8/bin/java\
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*'\
        -Xmx10g org.apache.spark.deploy.SparkSubmit\
        --master 'local[4]'\
        /ppml/trusted-big-data-ml/work/examples/start-fl-server.py -p 8981 --client-num=3" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee fl-server2.log

