#!/bin/bash
# At /ppml/trusted-big-data-ml

#set -x
# Modify Below Variables According To Your Environment
KMS_SERVER_IP=192.168.0.112
INPUT_DB_PATH=/ppml/trusted-big-data-ml/1m.db
LOCAL_IP=192.168.0.112

# Step 1. Generate Keys And Use Them to Encyypt Files Outside SGX With KMS
echo "[INFO] Start To Convert DB to CSVs And Encrypt..."
python /ppml/trusted-big-data-ml/work/kms-client/EncryptDBAutomation.py \
  --ip $KMS_SERVER_IP  --dbp $INPUT_DB_PATH
csv_path=$INPUT_DB_PATH.encrypted
echo "[INFO] The DB Is Transformed And Encrypted, Saved At $csv_path"

# Step 2. Decrypt The Encrypted Files As A Spark Job Inside SGX And Then Encrypt Columns
status_8_scala_e2e=1
echo "[INFO] Decrypt The Ciphere Files Inside SGX..."
if [ $status_8_scala_e2e -ne 0 ]; then
#SGX=1 ./pal_loader bash -c "bash ./work/kms-client/DecryptFilesWithSpark.sh $csv_path $KMS_SERVER_IP $LOCAL_IP" 2>&1 > spark-inside-sgx.log
bash ./work/kms-client/DecryptFilesWithSpark.sh $csv_path $KMS_SERVER_IP $LOCAL_IP
fi
status_8_scala_e2e=$(echo $?)

# Step 3. Decrypt The colums And Ouput With Kms API
echo "[INFO] Retrieve Output At Client Side."
echo "[INFO] Start To Decrypt Columns..."
python /ppml/trusted-big-data-ml/work/kms-client/DecryptColumnsWithCSV.py -ip 192.168.0.112 -path $csv_path.col_encrypted -pkp /ppml/trusted-big-data-ml/encrypted_primary_key -dkp /ppml/trusted-big-data-ml/encrypted_data_key
