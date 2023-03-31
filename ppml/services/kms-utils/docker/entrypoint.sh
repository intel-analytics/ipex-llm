# set -x
echo "PCCS_URL=$PCCS_URL" > /etc/sgx_default_qcnl.conf
echo "USE_SECURE_CERT=FALSE" >> /etc/sgx_default_qcnl.conf
action=$1
if [ "$action" = "enroll" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
		echo "ehsm does not support enroll by kms-utils, please enroll by calling rest API directly!"
	elif [ "$KMS_TYPE" = "simple" ]; then
		echo "Simple KMS is dummy. You can choose any appid and apikey. If you want to generate the corresponding primarykey and datakey, the appid must be 12 characters long."
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
		if [ ! -d "$BIGDL_HOME/python/bigdl" ]; then 
			unzip $BIGDL_HOME/python/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python/
		fi
		export APPID=$2
		export APIKEY=$3
		python3 $BIGDL_HOME/python/bigdl/ppml/kms/ehsm/client.py -api generate_primary_key -ip $EHSM_KMS_IP -port $EHSM_KMS_PORT
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
	    apikey=$3
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /home/key/simple_encrypted_primary_key \
		--dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType SimpleKeyManagementService \
		--simpleAPPID $appid \
		--simpleAPIKEY $apikey
	elif [ "$KMS_TYPE" = "azure" ]; then
	    keyVaultName=$2
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.GenerateKeys \
		--primaryKeyPath /home/key/simple_encrypted_primary_key \
		--dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType AzureKeyManagementService \
		--vaultName $keyVaultName
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
		return -1
	fi
elif [ "$action" = "encrypt" ]; then
	appid=$2
	apikey=$3
	input_path=$4
	data_source_type=$5
	if [ "$data_source_type" = "" ]; then 
		data_source_type=csv
	fi
	output_path=${input_path}.encrypted
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
	    apikey=$3
		/opt/jdk8/bin/java \
    		-cp "${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*:/${SPARK_HOME}/examples/jars/*:${BIGDL_HOME}/jars/*" -Xmx1g \
    		org.apache.spark.deploy.SparkSubmit \
    		--master local[2] \
        	--deploy-mode client \
			--driver-memory 5g \
			--driver-cores 4 \
			--executor-memory 5g \
			--executor-cores 4 \
			--num-executors 2 \
			--conf spark.cores.max=8 \
			--conf spark.network.timeout=10000000 \
			--conf spark.executor.heartbeatInterval=10000000 \
			--conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
			--conf spark.bigdl.primaryKey.amy.kms.type=EHSMKeyManagementService \
			--conf spark.bigdl.primaryKey.amy.material=/home/key/ehsm_encrypted_primary_key \
			--conf spark.bigdl.primaryKey.amy.kms.ip=$EHSM_KMS_IP \
			--conf spark.bigdl.primaryKey.amy.kms.port=$EHSM_KMS_PORT \
			--conf spark.bigdl.primaryKey.amy.kms.appId=$appid \
			--conf spark.bigdl.primaryKey.amy.kms.apiKey=$apikey \
			--verbose \
			--class com.intel.analytics.bigdl.ppml.utils.Encrypt \
			--conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
			--conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
			local://${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
			--inputDataSourcePath $input_path \
			--outputDataSinkPath $output_path \
			--cryptoMode aes/cbc/pkcs5padding \
			--dataSourceType $data_source_type
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=$2
	    apikey=$3
            input_path=$4
	    data_source_type=$5
	    if [ "$data_source_type" = "" ]; then 
	       data_source_type=csv
	    fi
	    output_path=${input_path}.encrypted
	    /opt/jdk8/bin/java \
		-cp "${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*:/${SPARK_HOME}/examples/jars/*:${BIGDL_HOME}/jars/*" -Xmx1g \
		org.apache.spark.deploy.SparkSubmit \
		--master local[2] \
		--deploy-mode client \
		--driver-memory 5g \
		--driver-cores 4 \
		--executor-memory 5g \
		--executor-cores 4 \
		--num-executors 2 \
		--conf spark.cores.max=8 \
		--conf spark.network.timeout=10000000 \
		--conf spark.executor.heartbeatInterval=10000000 \
		--conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
		--conf spark.bigdl.primaryKey.AmyPK.kms.type=SimpleKeyManagementService \
		--conf spark.bigdl.primaryKey.AmyPK.kms.appId=${appid} \
		--conf spark.bigdl.primaryKey.AmyPK.kms.apiKey=${apikey} \
		--conf spark.bigdl.primaryKey.AmyPK.material=/home/key/simple_encrypted_primary_key \
		--verbose \
		--class com.intel.analytics.bigdl.ppml.utils.Encrypt \
		--conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
		--conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
		local://${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
		--inputDataSourcePath $input_path \
		--outputDataSinkPath $output_path \
		--cryptoMode aes/cbc/pkcs5padding \
		--dataSourceType $data_source_type
    elif [ "$KMS_TYPE" = "azure" ]; then
        keyVaultName=$2
	/opt/jdk8/bin/java \
	  -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
	  com.intel.analytics.bigdl.ppml.examples.Encrypt \
          --inputPath $input_path \
          --primaryKeyPath /home/key/simple_encrypted_primary_key \
          --dataKeyPath /home/key/simple_encrypted_data_key \
           --kmsType AzureKeyManagementService \
	   --vaultName $keyVaultName
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
                return -1
        fi
elif [ "$action" = "encryptwithrepartition" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$2
            apikey=$3
	    input_path=$4
	    output_path=$input_path.encrypted
	    /opt/jdk8/bin/java \
	        -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
	        com.intel.analytics.bigdl.ppml.examples.EncryptWithRepartition   \
		--inputPath $input_path \
		--outputPath $output_path \
		--inputEncryptModeValue plain_text \
                --outputEncryptModeValue AES/CBC/PKCS5Padding \
		--outputPartitionNum 4 \
		--primaryKeyPath /home/key/ehsm_encrypted_primary_key \
		--dataKeyPath /home/key/ehsm_encrypted_data_key \
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
	    /opt/jdk8/bin/java \
		-cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.EncryptWithRepartition   \
		--inputPath $input_path \
		--outputPath $output_path \
		--inputEncryptModeValue plain_text \
                --outputEncryptModeValue AES/CBC/PKCS5Padding \
		--outputPartitionNum 4 \
                --primaryKeyPath /home/key/simple_encrypted_primary_key \
                --dataKeyPath /home/key/simple_encrypted_data_key \
		--kmsType SimpleKeyManagementService \
                --simpleAPPID $appid \
                --simpleAPIKEY $apikey
    elif [ "$KMS_TYPE" = "azure" ]; then
        keyVaultName=$2
        input_path=$3
	output_path=$input_path.encrypted
        /opt/jdk8/bin/java \
	  -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
	  com.intel.analytics.bigdl.ppml.examples.EncryptWithRepartition   \
	  --inputPath $input_path \
	  --outputPath $output_path \
	  --inputEncryptModeValue plain_text \
          --outputEncryptModeValue AES/CBC/PKCS5Padding \
	  --outputPartitionNum 4 \
          --primaryKeyPath /home/key/simple_encrypted_primary_key \
          --dataKeyPath /home/key/simple_encrypted_data_key \
	  --kmsType AzureKeyManagementService \
          --vaultName $keyVaultName
	else
           echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
           return -1
        fi
elif [ "$action" = "decrypt" ]; then
   if [ "$KMS_TYPE" = "ehsm" ]; then
      appid=$2
      apikey=$3
      input_path=$4
      data_source_type=$5
      if [ "$data_source_type" = "" ]; then 
         data_source_type=csv
      fi
      output_path=${input_path}.decrypt
      /opt/jdk8/bin/java \
        -cp "${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*:/${SPARK_HOME}/examples/jars/*:${BIGDL_HOME}/jars/*" -Xmx1g \
        org.apache.spark.deploy.SparkSubmit \
        --master local[2] \
        --driver-memory 5g \
        --driver-cores 4 \
        --executor-memory 5g \
        --executor-cores 4 \
        --num-executors 2 \
        --conf spark.cores.max=8 \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
        --conf spark.bigdl.primaryKey.amy.kms.type=EHSMKeyManagementService \
        --conf spark.bigdl.primaryKey.amy.material=/home/key/ehsm_encrypted_primary_key \
        --conf spark.bigdl.primaryKey.amy.kms.ip=$EHSM_KMS_IP \
        --conf spark.bigdl.primaryKey.amy.kms.port=$EHSM_KMS_PORT \
        --conf spark.bigdl.primaryKey.amy.kms.appId=$appid \
        --conf spark.bigdl.primaryKey.amy.kms.apiKey=$apikey \
        --verbose \
        --class com.intel.analytics.bigdl.ppml.utils.Encrypt \
        --conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
        --conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
        --name decrypt \
        local://${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
        --inputDataSourcePath $input_path \
        --outputDataSinkPath $output_path \
        --cryptoMode aes/cbc/pkcs5padding \
        --dataSourceType $data_source_type \
        --action decrypt
   elif [ "$KMS_TYPE" = "simple" ]; then
	appid=$2
        apikey=$3
        input_path=$4
	data_source_type=$5
	if [ "$data_source_type" = "" ]; then 
	   data_source_type=csv
	fi
	output_path=${input_path}.decrypt
	/opt/jdk8/bin/java \
	    -cp "${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*:/${SPARK_HOME}/examples/jars/*:${BIGDL_HOME}/jars/*" -Xmx1g \
	    org.apache.spark.deploy.SparkSubmit \
            --master local[2] \
            --driver-memory 5g \
            --driver-cores 4 \
            --executor-memory 5g \
            --executor-cores 4 \
	    --num-executors 2 \
	    --conf spark.cores.max=8 \
	    --conf spark.network.timeout=10000000 \
	    --conf spark.executor.heartbeatInterval=10000000 \
	    --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
	    --conf spark.bigdl.primaryKey.AmyPK.kms.type=SimpleKeyManagementService \
	    --conf spark.bigdl.primaryKey.AmyPK.kms.appId=${appid} \
	    --conf spark.bigdl.primaryKey.AmyPK.kms.apiKey=${apikey} \
	    --conf spark.bigdl.primaryKey.AmyPK.material=/home/key/simple_encrypted_primary_key \
	    --verbose \
	    --class com.intel.analytics.bigdl.ppml.utils.Encrypt \
	    --conf spark.executor.extraClassPath=$BIGDL_HOME/jars/* \
	    --conf spark.driver.extraClassPath=$BIGDL_HOME/jars/* \
	    --name amy-encrypt \
	    local://${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar \
	    --inputDataSourcePath $input_path \
	    --outputDataSinkPath $output_path \
	    --cryptoMode aes/cbc/pkcs5padding \
	    --dataSourceType csv \
	    --action decrypt
    elif [ "$KMS_TYPE" = "azure" ]; then
        keyVaultName=$2
        input_path=$3
	/opt/jdk8/bin/java \
	   -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
	   com.intel.analytics.bigdl.ppml.examples.Decrypt \
           --inputPath $input_path \
           --inputPartitionNum 8 \
           --outputPartitionNum 8 \
           --inputEncryptModeValue AES/CBC/PKCS5Padding \
           --outputEncryptModeValue plain_text \
           --primaryKeyPath /home/key/simple_encrypted_primary_key \
           --dataKeyPath /home/key/simple_encrypted_data_key \
           --kmsType AzureKeyManagementService \
           --vaultName $keyVaultName
	else
           echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple, (3) azure"
           return -1
        fi
else
   echo "Wrong action! Action can be (1) enroll, (2) generatekeys, (3) encrypt, (4) decrypt, and (5) encryptwithrepartition."
fi
