#!/bin/bash

bash ppml-spark-submit.sh \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
        --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class com.intel.analytics.bigdl.models.lenet.Train \
        --driver-memory 10G \
        /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
        -f /ppml/trusted-big-data-ml/work/data \
        -b 64 \
        -e 1 | tee spark.local.sgx.log
