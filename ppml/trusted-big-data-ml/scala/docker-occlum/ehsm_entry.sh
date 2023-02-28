# set -x
rm /etc/sgx_default_qcnl.conf
#echo "PCCS_URL=$PCCS_URL" > /etc/sgx_default_qcnl.conf
echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
echo "USE_SECURE_CERT=FALSE" >> /etc/sgx_default_qcnl.conf
action=$1
EHSM_URL=${ATTESTATION_URL}
EHSM_KMS_IP=${EHSM_URL%:*}
EHSM_KMS_PORT=${EHSM_URL#*:}
mkdir -p /opt/occlum_spark/data/key/
mkdir -p /opt/occlum_spark/data/encrypt/
export KMS_TYPE=ehsm
#only support ehsm now
if [ "$action" = "generatekeys" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
	    apikey=$3
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
		--kmsServerPort $EHSM_KMS_PORT \
		--ehsmAPPID $appid \
		--ehsmAPIKEY $apikey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
	    apikey=$3
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /home/key/simple_encrypted_primary_key \
		--kmsType SimpleKeyManagementService \
		--simpleAPPID $appid \
		--simpleAPIKEY $apikey
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
		return -1
	fi
elif [ "$action" = "encrypt" ]; then
	appid=$2
	apikey=$3
	input_path=$4
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
	    apikey=$3
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.Encrypt \
		--inputPath $input_path \
		--primaryKeyPath /opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
                --kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
                --kmsServerPort $EHSM_KMS_PORT \
                --ehsmAPPID $appid \
                --ehsmAPIKEY $apikey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
	    apikey=$3
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.Encrypt \
                --inputPath $input_path \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --kmsType SimpleKeyManagementService \
		--simpleAPPID $appid \
                --simpleAPIKEY $apikey
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
                return -1
        fi
elif [ "$action" = "encryptwithrepartition" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
            apikey=$3
	    input_path=$4
	    output_path=$input_path.encrypted
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.EncryptWithRepartition   \
		--inputPath $input_path \
		--outputPath $output_path \
		--inputEncryptModeValue plain_text \
                --outputEncryptModeValue AES/CBC/PKCS5Padding \
		--outputPartitionNum 4 \
		--primaryKeyPath /opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
                --kmsServerPort $EHSM_KMS_PORT \
                --ehsmAPPID $appid \
                --ehsmAPIKEY $apikey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
            apikey=$3
	    input_path=$4
	    output_path=$input_path.encrypted
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.EncryptWithRepartition   \
		--inputPath $input_path \
		--outputPath $output_path \
		--inputEncryptModeValue plain_text \
                --outputEncryptModeValue AES/CBC/PKCS5Padding \
		--outputPartitionNum 4 \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
		--kmsType SimpleKeyManagementService \
                --simpleAPPID $appid \
                --simpleAPIKEY $apikey
	else
                echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
                return -1
        fi
elif [ "$action" = "decrypt" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	appid=$2
        apikey=$3
        input_path=$4
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.Decrypt \
		--inputPath $input_path \
		--inputPartitionNum 8 \
		--outputPartitionNum 8 \
		--inputEncryptModeValue AES/CBC/PKCS5Padding \
		--outputEncryptModeValue plain_text \
		--primaryKeyPath /opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
                --kmsServerPort $EHSM_KMS_PORT \
                --ehsmAPPID $appid \
                --ehsmAPIKEY $apikey
	elif [ "$KMS_TYPE" = "simple" ]; then
	appid=$2
        apikey=$3
        input_path=$4
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.Decrypt \
                --inputPath $input_path \
                --inputPartitionNum 8 \
                --outputPartitionNum 8 \
                --inputEncryptModeValue AES/CBC/PKCS5Padding \
                --outputEncryptModeValue plain_text \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --kmsType SimpleKeyManagementService \
                --simpleAPPID $appid \
                --simpleAPIKEY $apikey
	else
                echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
                return -1
        fi
else
	echo "Wrong action! Action can be (1) generatekeys, (2) encrypt, (3) decrypt, and (4) encryptwithrepartition."
fi
