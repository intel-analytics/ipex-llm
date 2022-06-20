# set -x
echo "PCCS_URL=$PCCS_URL" > /etc/sgx_default_qcnl.conf
echo "USE_SECURE_CERT=FALSE" >> /etc/sgx_default_qcnl.conf
action=$1
if [ "$action" = "enroll" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
		cd /home/ehsm/out/ehsm-kms_enroll_app/
		./ehsm-kms_enroll_app -a http://$EHSM_KMS_IP:$EHSM_KMS_PORT/ehsm/
	elif [ "$KMS_TYPE" = "simple" ]; then
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.GenerateSimpleAppidAndAppkey
	elif [ "$KMS_TYPE" = "azure" ]; then
	    keyVaultName=$2
	    id=$3
		az keyvault set-policy --name $keyVaultName --object-id $id \
		--secret-permissions all --key-permissions all --certificate-permissions all
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
		return -1
	fi
elif [ "$action" = "generatekeys" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
	    appkey=$3
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /home/key/ehsm_encrypted_primary_key \
		--dataKeyPath /home/key/ehsm_encrypted_data_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
		--kmsServerPort $EHSM_KMS_PORT \
		--ehsmAPPID $appid \
		--ehsmAPPKEY $appkey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
	    appkey=$3
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /home/key/simple_encrypted_primary_key \
		--dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType SimpleKeyManagementService \
		--simpleAPPID $appid \
		--simpleAPPKEY $appkey
	elif [ "$KMS_TYPE" = "azure" ]; then
	    keyVaultName=$2
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /home/key/simple_encrypted_primary_key \
		--dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType AzureKeyManagementService \
		--vaultName $keyVaultName
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
		return -1
	fi
elif [ "$action" = "localcrypto" ]; then
	appid=$2
	appkey=$3
	input_path=$4
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
	    appkey=$3
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.LocalCryptoExample \
		--inputPath $input_path \
		--primaryKeyPath /home/key/ehsm_encrypted_primary_key \
                --dataKeyPath /home/key/ehsm_encrypted_data_key \
                --kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
                --kmsServerPort $EHSM_KMS_PORT \
                --ehsmAPPID $appid \
                --ehsmAPPKEY $appkey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
	    appkey=$3
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.LocalCryptoExample \
                --inputPath $input_path \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
                --kmsType SimpleKeyManagementService \
		--simpleAPPID $appid \
                --simpleAPPKEY $appkey
    elif [ "$KMS_TYPE" = "azure" ]; then
        keyVaultName=$2
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.LocalCryptoExample \
                --inputPath $input_path \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
                --kmsType AzureKeyManagementService \
		        --vaultName $keyVaultName
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
                return -1
        fi
elif [ "$action" = "splitandencrypt" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
        appkey=$3
	    input_path=$4
	    output_path=$input_path.encrypted
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.SplitAndEncrypt   \
		--inputPath $input_path \
		--outputPath $output_path \
		--outputPartitionNum 4 \
		--outputCryptoModeValue AES/CBC/PKCS5Padding \
		--primaryKeyPath /home/key/ehsm_encrypted_primary_key \
		--dataKeyPath /home/key/ehsm_encrypted_data_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
                --kmsServerPort $EHSM_KMS_PORT \
                --ehsmAPPID $appid \
                --ehsmAPPKEY $appkey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
        appkey=$3
	    input_path=$4
	    output_path=$input_path.encrypted
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.SplitAndEncrypt   \
		--inputPath $input_path \
                --outputPath $output_path \
                --outputPartitionNum 4 \
                --outputCryptoModeValue AES/CBC/PKCS5Padding \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType SimpleKeyManagementService \
                --simpleAPPID $appid \
                --simpleAPPKEY $appkey
    elif [ "$KMS_TYPE" = "azure" ]; then
        keyVaultName=$2
        input_path=$3
	    output_path=$input_path.encrypted
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.SplitAndEncrypt   \
		--inputPath $input_path \
                --outputPath $output_path \
                --outputPartitionNum 4 \
                --outputCryptoModeValue AES/CBC/PKCS5Padding \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType AzureKeyManagementService \
                --vaultName $keyVaultName
	else
                echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
                return -1
        fi
elif [ "$action" = "simplequery" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
        appkey=$3
        input_path=$4
        output_path=$input_path.decrypted
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.SimpleQueryExample \
		--inputPath $input_path \
		--outputPath $output_path \
		--inputPartitionNum 8 \
		--outputPartitionNum 8 \
		--inputCryptoModeValue AES/CBC/PKCS5Padding \
		--outputCryptoModeValue plain_text \
		--primaryKeyPath /home/key/ehsm_encrypted_primary_key \
		--dataKeyPath /home/key/ehsm_encrypted_data_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
                --kmsServerPort $EHSM_KMS_PORT \
                --ehsmAPPID $appid \
                --ehsmAPPKEY $appkey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
        appkey=$3
        input_path=$4
        output_path=$input_path.decrypted
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.SimpleQueryExample \
                --inputPath $input_path \
                --outputPath $output_path \
                --inputPartitionNum 8 \
                --outputPartitionNum 8 \
                --inputCryptoModeValue AES/CBC/PKCS5Padding \
                --outputCryptoModeValue plain_text \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
                --kmsType SimpleKeyManagementService \
                --simpleAPPID $appid \
                --simpleAPPKEY $appkey
    elif [ "$KMS_TYPE" = "azure" ]; then
        keyVaultName=$2
        input_path=$3
        output_path=$input_path.decrypted
		java -cp /home/spark-encrypt-io.jar \
		com.intel.analytics.bigdl.ppml.examples.SimpleQueryExample \
                --inputPath $input_path \
                --outputPath $output_path \
                --inputPartitionNum 8 \
                --outputPartitionNum 8 \
                --inputCryptoModeValue AES/CBC/PKCS5Padding \
                --outputCryptoModeValue plain_text \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
                --kmsType AzureKeyManagementService \
                --vaultName $keyVaultName
	else
                echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
                return -1
        fi
else
	echo "Wrong action! Action can be (1) enroll, (2) generatekeys, (3) localcrypto, (4) splitandencrypt, and (5) simplequery."
fi
