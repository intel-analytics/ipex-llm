#!/bin/bash/ppml/trusted-big-data-ml/1m_csvINPUT_DIR_PATH
# At /ppml/trusted-big-data-ml

# set -x
# Modify Below Variables According To Your Environment
KMS_SERVER_IP=192.168.0.112 # KMS Server IP
INPUT_DIR_PATH=/ppml/trusted-big-data-ml/1m_csv # Path Of The Directory Containing CSVs
LOCAL_IP=192.168.0.112 # Spark Host IP

# Step 1. Generate Primary Key And Data Key
echo "[INFO] Start To Generate Keys..."
python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py \
  --api generate_primary_key --ip $KMS_SERVER_IP
python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py \
  --api generate_data_key --ip $KMS_SERVER_IP --pkp /ppml/trusted-big-data-ml/encrypted_primary_key

# Step 2. Encyypt The Input Directory Outside SGX With KMS
echo "[INFO] Start To Encrypt..."
python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py \
  --api encrypt_directory_with_key --ip $KMS_SERVER_IP  --dir $INPUT_DIR_PATH \
  --pkp /ppml/trusted-big-data-ml/encrypted_primary_key --dkp /ppml/trusted-big-data-ml/encrypted_data_key
encrypted_path=$INPUT_DIR_PATH.encrypted
echo "[INFO] The Directory Is Encrypted, Saved At $encrypted_path"

# Step 3. Decrypt The Encrypted Files As A Spark Job Inside SGX And Then Encrypt Columns
status_8_scala_e2e=1
echo "[INFO] Decrypt The Ciphere Files Inside SGX..."
if [ $status_8_scala_e2e -ne 0 ]; then
SGX=1 ./pal_loader bash -c "bash ./work/kms-client/DecryptFilesWithSpark.sh $encrypted_path $KMS_SERVER_IP $LOCAL_IP" 2>&1 > spark-inside-sgx.log
fi
status_8_scala_e2e=$(echo $?)
output_path=$encrypted_path.col_encrypted
echo "[INFO] The Output is Saved At $output_path"

# Step 4. Decrypt The colums And Ouput With KMS API
echo "[INFO] Retrieve Output At Client Side."
echo "[INFO] Start To Decrypt Columns..."
python /ppml/trusted-big-data-ml/work/kms-client/KMS_Client.py --api decrypt_csv_columns --ip $KMS_SERVER_IP --dir $output_path --pkp /ppml/trusted-big-data-ml/encrypted_primary_key --dkp /ppml/trusted-big-data-ml/encrypted_data_key
