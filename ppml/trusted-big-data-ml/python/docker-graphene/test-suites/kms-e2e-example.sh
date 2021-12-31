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
if [ $status_8_scala_e2e -ne 0 ]; then
SGX=1 ./pal_loader bash -c "bash ./work/kms-client/DecryptFilesWithSpark.sh $KEYWHIZ_SERVER_IP $LOCAL_IP" 2>&1 > spark-inside-sgx.log
fi
status_8_scala_e2e=$(echo $?)

echo "[INFO] The Result Of Decrypted CSV Files Content Queried By Spark Is As Below:"
cat spark-inside-sgx.log | grep "|"
