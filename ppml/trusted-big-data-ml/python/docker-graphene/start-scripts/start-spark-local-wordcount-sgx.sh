#!/bin/bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
    -Xmx1g org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    /ppml/trusted-big-data-ml/work/spark-2.4.3/examples/src/main/python/wordcount.py ./work/examples/helloworld.py" | tee test-wordcount-sgx.log
