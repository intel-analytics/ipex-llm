#set -x
SPARK_EXTRA_JAR_PATH=
SPARK_JOB_MAIN_CLASS=
ARGS=
RUNTIME_SPARK_MASTER=
DATA_LAKE_NAME=
DATA_LAKE_ACCESS_KEY=
KEY_VAULT_NAME=
PRIMARY_KEY_PATH=
DATA_KEY_PATH=

INPUT_DIR_PATH=$1
ENCRYPT_KEYS_PATH=$2
OUTPUT_DIR_PATH=$3
LOCAL_IP=$4

secure_password=`az keyvault secret show --name "key-pass" --vault-name $KEY_VAULT_NAME --query "value" | sed -e 's/^"//' -e 's/"$//'`

export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx12g \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-decrypt-sgx \
    --conf spark.driver.memory=18g \
    --conf spark.driver.cores=2 \
    --conf spark.executor.cores=2 \
    --conf spark.executor.memory=24g \
    --conf spark.executor.instances=1 \
    --conf spark.driver.defaultJavaOptions="-Dlog4j.configuration=/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/log4j2.xml" \
    --conf spark.executor.defaultJavaOptions="-Dlog4j.configuration=/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/log4j2.xml" \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT \
    --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/azure/spark-driver-template-az.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/azure/spark-executor-template-az.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.sql.auto.repartition=true \
    --conf spark.default.parallelism=400 \
    --conf spark.sql.shuffle.partitions=400 \
    --jars local://$SPARK_EXTRA_JAR_PATH \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.driver.mem=16g \
    --conf spark.kubernetes.sgx.driver.jvm.mem=7g \
    --conf spark.kubernetes.sgx.executor.mem=16g \
    --conf spark.kubernetes.sgx.executor.jvm.mem=7g \
    --conf spark.kubernetes.sgx.log.level=error \
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
    --conf spark.hadoop.fs.azure.account.auth.type.${DATA_LAKE_NAME}.dfs.core.windows.net=SharedKey \
    --conf spark.hadoop.fs.azure.account.key.${DATA_LAKE_NAME}.dfs.core.windows.net=${DATA_LAKE_ACCESS_KEY} \
    --conf spark.hadoop.fs.azure.enable.append.support=true \
    --conf spark.bigdl.kms.type=AzureKeyManagementService \
    --conf spark.bigdl.kms.azure.vault=$KEY_VAULT_NAME \
    --conf spark.bigdl.kms.key.primary=$PRIMARY_KEY_PATH \
    --conf spark.bigdl.kms.key.data=$DATA_KEY_PATH \
    --class $SPARK_JOB_MAIN_CLASS \
    --verbose \
    $SPARK_EXTRA_JAR_PATH \
    $ARGS
