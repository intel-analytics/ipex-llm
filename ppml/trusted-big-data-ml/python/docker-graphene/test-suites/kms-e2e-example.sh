#!/bin/bash
# At /ppml/trusted-big-data-ml

#set -x
KEYWHIZ_SERVER_IP=192.168.0.112
LOCAL_IP=192.168.0.112
INPUT_DIR_PATH=/ppml/trusted-big-data-ml/work/input
ENCRYPTED_SAVE_DIR_PATH=/ppml/trusted-big-data-ml/work/encrypted_output

# Step 1. Data Preparation
mkdir $INPUT_DIR_PATH
cp /ppml/trusted-big-data-ml/work/data/kms-example/* $INPUT_DIR_PATH
mkdir $ENCRYPTED_SAVE_DIR_PATH

# Step 2. Generate Keys And Use Them to Encyypt Files Outside SGX With KMS 
echo "[INFO] Start To Process Outside SGX..."
# Files Under INPUT_DIR_PATH Will Be Encrypted And Saved In SAVE_DIR_PATH
python ./work/kms-client/EncryptFilesAutomation.py -ip $KEYWHIZ_SERVER_IP -input_dir $INPUT_DIR_PATH -save_dir $ENCRYPTED_SAVE_DIR_PATH
echo "[INFO] Encrypted Files Are Saved Under $ENCRYPTED_SAVE_DIR_PATH."

# Step 3. Decrypt The Encrypted File As A Spark Job Inside SGX
status_8_scala_e2e=1
echo "[INFO] Decrypt The Ciphere Files Inside SGX..."
<<<<<<< HEAD:ppml/trusted-big-data-ml/python/docker-graphene/test-suites/kms-e2e-example.sh
if [ $status_8_scala_e2e -ne 0 ]; then
SGX=1 ./pal_loader bash -c "bash ./work/kms-client/DecryptFilesWithSpark.sh $KEYWHIZ_SERVER_IP $LOCAL_IP" 2>&1 > spark-inside-sgx.log
fi
status_8_scala_e2e=$(echo $?)
||||||| merged common ancestors
SGX=1 ./pal_loader bash -c "$SPARK_HOME/bin/spark-submit \
  --master local[2] \
  --class sparkDecryptFiles.decryptFiles \
  --jars /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/fernet-java8-1.4.2.jar \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/sparkdecryptfiles_2.12-0.1.0.jar \
  $ENCRYPTED_SAVE_DIR_PATH \
  Fernet $(python ./work/kms-client/GetDataKeyPlaintext.py -ip $KEYWHIZ_SERVER_IP -pkp ./encrypted_primary_key -dkp ./encrypted_data_key)" 2>&1 > spark-inside-sgx.log
=======
if [ $status_8_scala_e2e -ne 0 ]; then
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-example-sql-e2e.jar' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master local[2] \
  --conf spark.driver.host=$LOCAL_IP \
  --conf spark.driver.memory=8g \
  --executor-memory 8g \
  --class sparkDecryptFiles.decryptFiles \
  --jars /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/fernet-java8-1.4.2.jar \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/sparkdecryptfiles_2.12-0.1.0.jar \
  $ENCRYPTED_SAVE_DIR_PATH \
  Fernet $(python ./work/kms-client/GetDataKeyPlaintext.py -ip $KEYWHIZ_SERVER_IP -pkp ./encrypted_primary_key -dkp ./encrypted_data_key)" 2>&1 > spark-inside-sgx.log
fi
status_8_scala_e2e=$(echo $?)
>>>>>>> f649678ae9250e5500158f989b79d37bb8690b04:ppml/trusted-big-data-ml/python/docker-graphene/test-suites/kms-saprk-e2e-example.sh

echo "[INFO] The Result Of Decrypted CSV Files Content Queried By Spark Is As Below:"
cat spark-inside-sgx.log | grep "|"
