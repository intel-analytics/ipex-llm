#!/bin/bash
rm /ppml/data/weekly-test/tpc-h/TIMES.txt

export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 1g\
        --sgx-executor-jvm-memory 3g\
        --num-executors 8 \
        --driver-memory 1g \
        --driver-cores 8 \
        --executor-memory 1g \
        --executor-cores 8\
        --conf spark.cores.max=64 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.container.image.pullPolicy=Always \
        --class com.intel.analytics.bigdl.ppml.examples.tpch.TpchQuery \
        --conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
        --conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
        --conf spark.kubernetes.file.upload.path=file:///tmp \
        --name weekly-tpc-h-on-gramine \
        --verbose \
        local://$BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
        hdfs://172.168.0.206:9000/tpc-h/1g-native \
        /ppml/data/weekly-test/tpc-h \
        plain_text \
        plain_text

if [ -f "/ppml/data/weekly-test/tpc-h/TIMES.txt" ]; then
        echo "TPC-H test success!"
else
        echo "TPC-H test failed!"
        exit 1
fi