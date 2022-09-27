#!/bin/bash
cd /ppml/trusted-big-data-ml
./clean.sh
gramine-argv-serializer bash -c "/opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx1g org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/pi.py" > secured_argvs
./init.sh
gramine-sgx bash 2>&1 | tee test-pi-sgx.log
cat /ppml/trusted-big-data-ml/test-pi-sgx.log | egrep "roughly"