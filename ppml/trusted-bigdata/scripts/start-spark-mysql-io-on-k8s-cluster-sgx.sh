export mysql_ip=your_mysql_ip
export mysql_db=your_mysql_db

export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 1g\
        --sgx-executor-jvm-memory 3g\
        --num-executors 2 \
        --driver-memory 1g \
        --driver-cores 8 \
        --executor-memory 1g \
        --executor-cores 8\
        --conf spark.cores.max=16 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --class com.intel.analytics.bigdl.ppml.examples.MySQLSparkExample \
        --name spark-mysql-io \
        --conf spark.kubernetes.file.upload.path=file:///tmp \
        --verbose \
        --log-file spark-mysql-io-sgx-cluster.log \
        --jars local:///ppml/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar \
        local:///ppml/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar $mysql_ip/$mysql_db