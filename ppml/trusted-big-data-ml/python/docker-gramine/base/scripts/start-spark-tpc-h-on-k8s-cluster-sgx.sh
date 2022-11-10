#!/bin/bash
rm /ppml/trusted-big-data-ml/work/data/weekly-test/tpc-h/TIMES.txt

export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 20g\
        --sgx-executor-jvm-memory 10g\
        --num-executors 8 \
        --driver-memory 20g \
        --driver-cores 8 \
        --executor-memory 10g \
        --executor-cores 8\
        --conf spark.cores.max=72 \
        --conf spark.kubernetes.driver.container.image=10.239.45.10/arda/intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference-64g:2.2.0-SNAPSHOT \
        --conf spark.kubernetes.executor.container.image=10.239.45.10/arda/intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference-32g:2.2.0-SNAPSHOT \
        --conf spark.kubernetes.container.image.pullPolicy=Always \
        --class com.intel.analytics.bigdl.ppml.examples.tpch.TpchQuery \
        --conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
        --conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
        --conf spark.kubernetes.file.upload.path=file:///tmp \
        --conf spark.kubernetes.node.selector.label=dev \
        --name weekly-tpc-h-on-gramine \
        --verbose \
        local://$BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
        hdfs://172.168.0.206:9000/tpc-h/1g-native \
        /ppml/trusted-big-data-ml/work/data/weekly-test/tpc-h \
        plain_text \
        plain_text

if [ -f "/ppml/trusted-big-data-ml/work/data/weekly-test/tpc-h/TIMES.txt" ]; then
        echo "TPC-H test success!"
else
        echo "TPC-H test failed!"
        exit 1
fi