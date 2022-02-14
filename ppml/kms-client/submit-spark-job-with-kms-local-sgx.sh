# set -x
SPARK_EXTRA_JAR_PATH=/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-encrypt-io-0.1-SNAPSHOT.jar
SPARK_JOB_MAIN_CLASS=com.intel.analytics.bigdl.ppml.e2e.examples.SimpleEncryptIO
INPUT_DIR_PATH=$1
ENCRYPT_KEYS_PATH=$2
OUTPUT_DIR_PATH=$3
KMS_SERVER_IP=$4
KMS_SERVER_PORT=$5
LOCAL_IP=$6
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt < /ppml/trusted-big-data-ml/work/password/output.bin`

SGX=1 ./pal_loader bash -c "\
/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-example-sql-e2e.jar' \
  -Xmx12g \
  org.apache.spark.deploy.SparkSubmit \
  --master local[2] \
  --conf spark.driver.host=$LOCAL_IP \
  --conf spark.driver.memory=16g \
  --num-executors 10 \
  --executor-cores 8 \
  --executor-memory 16g \
  --conf spark.ssl.enabled=true \
  --conf spark.ssl.port=8043 \
  --conf spark.ssl.keyPassword=$secure_password \
  --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
  --conf spark.ssl.keyStorePassword=$secure_password \
  --conf spark.ssl.keyStoreType=JKS \
  --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
  --conf spark.ssl.trustStorePassword=$secure_password \
  --conf spark.ssl.trustStoreType=JKS \
  --class $SPARK_JOB_MAIN_CLASS \
  $SPARK_EXTRA_JAR_PATH \
  $INPUT_DIR_PATH \
  $KMS_SERVER_IP \
  $KMS_SERVER_PORT \
  $ENCRYPT_KEYS_PATH \
  $OUTPUT_DIR_PATH" 2>&1 | tee spark-decrypt-local-sgx.log
