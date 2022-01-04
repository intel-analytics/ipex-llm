set -x
FERNET_JAR_PATH=/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/fernet-java8-1.4.2.jar
SPARK_DECRYPT_JAR_PATH=/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/sparkdecryptfiles_2.12-0.1.0.jar
ENCRYPTED_SAVE_DIR_PATH=/ppml/trusted-big-data-ml/work/encrypted_output
KEYWHIZ_SERVER_IP=$1
KMS_CLIENT_DIR_PAHT=/ppml/trusted-big-data-ml/work/kms-client
LOCAL_IP=$2
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`

/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-example-sql-e2e.jar' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master local[2] \
  --conf spark.driver.host=$LOCAL_IP \
  --conf spark.driver.memory=8g \
  --conf spark.ssl.enabled=true \
  --conf spark.ssl.port=8043 \
  --conf spark.ssl.keyPassword=$secure_password \
  --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
  --conf spark.ssl.keyStorePassword=$secure_password \
  --conf spark.ssl.keyStoreType=JKS \
  --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
  --conf spark.ssl.trustStorePassword=$secure_password \
  --conf spark.ssl.trustStoreType=JKS \
  --executor-memory 8g \
  --class sparkDecryptFiles.decryptFiles \
  --jars $FERNET_JAR_PATH \
  $SPARK_DECRYPT_JAR_PATH \
  $ENCRYPTED_SAVE_DIR_PATH \
  Fernet $(python $KMS_CLIENT_DIR_PAHT/GetDataKeyPlaintext.py -ip $KEYWHIZ_SERVER_IP -pkp ./encrypted_primary_key -dkp ./encrypted_data_key)
