# tpch-spark

TPC-H queries implemented in Spark using the DataFrames API running with BigDL PPML.

## Generating tables

Go to [TPC Download](https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) site, choose `TPC-H` source code, then download the TPC-H toolkits.
After you download the tpc-h tools zip and uncompressed the zip file. Go to `dbgen` directory, and create a makefile based on `makefile.suite`, and run `make`.

This should generate an executable called `dbgen`
```
./dbgen -h
```

gives you the various options for generating the tables. The simplest case is running:
```
./dbgen
```
which generates tables with extension `.tbl` with scale 1 (default) for a total of rougly 1GB size across all tables. For different size tables you can use the `-s` option:
```
./dbgen -s 10
```
will generate roughly 10GB of input data.

You can then either upload your data to remote file system or read them locally.

## Encrypt Data
Encrypt data with specified Key Management Service (`SimpleKeyManagementService`, or `EHSMKeyManagementService` , or `AzureKeyManagementService`)

The example code of encrypt data with `SimpleKeyManagementService` is like below:
```
export BIGDL_HOME=XXX
java -cp '$BIGDL_HOME/lib/bigdl-ppml-VERSION-jar-with-dependencies.jar \
   -Xmx10g \
   com.intel.analytics.bigdl.ppml.examples.tpch.EncryptFiles \
   --kmsType SimpleKeyManagementService \
   --simpleAPPID xxxxxxxxxxxx \
   --simpleAPIKEY xxxxxxxxxxxx \
   --inputPath xxx/dbgen \
   --outputPath xxx/dbgen-encrypted
```

## Running

Make sure you set the INPUT_DIR and OUTPUT_DIR in `TpchQuery` class before compiling to point to the
location the of the input data and where the output should be saved.

The example script to run a query is like:

```
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin` && \
export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
export SPARK_LOCAL_IP=$LOCAL_IP && \
export RUNTIME_SPARK_MASTER=xxx && \
export INPUT_DIR=xxx/dbgen && \
export OUTPUT_DIR=xxx/output && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/lib/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    -Dbigdl.mklNumThreads=1 \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-tpch-sgx \
    --conf spark.driver.host=$LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=10g \
    --conf spark.driver.blockManager.port=10026 \
    --conf spark.blockManager.port=10025 \
    --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
    --conf spark.worker.timeout=600 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.network.timeout=10000000 \
    --conf spark.starvation.timeout=250000 \
    --conf spark.rpc.askTimeout=600 \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --conf spark.io.compression.codec=lz4 \
    --conf spark.sql.shuffle.partitions=8 \
    --conf spark.speculation=false \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.executor.instances=24 \
    --executor-cores 8 \
    --total-executor-cores 192 \
    --executor-memory 16G \
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
    --conf spark.kubernetes.authenticate.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.executor.podNamePrefix=spark-tpch-sgx \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.executor.mem=32g \
    --conf spark.kubernetes.sgx.executor.jvm.mem=10g \
    --conf spark.kubernetes.sgx.log.level=$SGX_LOG_LEVEL \
    --conf spark.authenticate=true \
    --conf spark.authenticate.secret=$secure_password \
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
    --conf spark.authenticate.enableSaslEncryption=true \
    --conf spark.network.crypto.enabled=true \
    --conf spark.network.crypto.keyLength=128 \
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    --conf spark.io.encryption.enabled=true \
    --conf spark.io.encryption.keySizeBits=128 \
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1 \
    --conf spark.ssl.enabled=true \
    --conf spark.ssl.port=8043 \
    --conf spark.ssl.keyPassword=$secure_password \
    --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.keyStorePassword=$secure_password \
    --conf spark.ssl.keyStoreType=JKS \
    --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.trustStorePassword=$secure_password \
    --conf spark.ssl.trustStoreType=JKS \
    --conf spark.bigdl.kms.type=SimpleKeyManagementService \
    --conf spark.bigdl.kms.simple.id=simpleAPPID \
    --conf spark.bigdl.kms.simple.key=simpleAPIKEY \
    --conf spark.bigdl.kms.key.primary=xxxx/primaryKey \
    --conf spark.bigdl.kms.key.data=xxxx/dataKey \
    --class com.intel.analytics.bigdl.ppml.examples.tpch.TpchQuery \
    --verbose \
    /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/lib/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
    $INPUT_DIR $OUTPUT_DIR aes_cbc_pkcs5padding plain_text [QUERY]
```

INPUT_DIR is the tpch's data dir.
OUTPUT_DIR is the dir to write the query result.
The optional parameter [QUERY] is the number of the query to run e.g 1, 2, ..., 22

----------------
This project is based on [Savvas Savvides's tpch-spark](https://github.com/ssavvides/tpch-spark).

