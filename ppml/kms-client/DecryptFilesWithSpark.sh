# set -x
SPARK_DECRYPT_JAR_PATH=/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/sparkcryptofiles-1.0-SNAPSHOT.jar
CSV_DIR_PATH=$1
KMS_SERVER_IP=$2
LOCAL_IP=$3
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
data_key=$(python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py -api get_data_key_plaintext -ip $KMS_SERVER_IP -pkp ./encrypted_primary_key -dkp ./encrypted_data_key)

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
  --class sparkCryptoFiles.encryptColumnFromDifferentTables \
  $SPARK_DECRYPT_JAR_PATH \
  $CSV_DIR_PATH \
  $data_key \
  $CSV_DIR_PATH.col_encrypted
