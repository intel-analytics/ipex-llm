#!/bin/bash
bash ppml-spark-submit.sh \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class org.apache.spark.examples.SparkPi \
        --driver-memory 10G \
        /ppml/trusted-big-data-ml/work/spark-2.4.6/examples/jars/spark-examples_2.11-2.4.6.jar | tee spark.local.pi.sgx.log
