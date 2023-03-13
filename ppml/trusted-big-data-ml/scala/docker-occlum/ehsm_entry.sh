# set -x
rm /etc/sgx_default_qcnl.conf
#echo "PCCS_URL=$PCCS_URL" > /etc/sgx_default_qcnl.conf
echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
echo "USE_SECURE_CERT=FALSE" >> /etc/sgx_default_qcnl.conf
action=$1
KMS_TYPE=$2
EHSM_URL=${ATTESTATION_URL}
EHSM_KMS_IP=${EHSM_URL%:*}
EHSM_KMS_PORT=${EHSM_URL#*:}
mkdir -p /opt/occlum_spark/data/key/
mkdir -p /opt/occlum_spark/data/encrypt/
#only support ehsm now
if [ "$action" = "generatekey" ]; then
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$3
	    apikey=$4
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.GeneratePrimaryKey \
		--primaryKeyPath /opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
		--kmsType EHSMKeyManagementService \
		--kmsServerIP $EHSM_KMS_IP \
		--kmsServerPort $EHSM_KMS_PORT \
		--ehsmAPPID $appid \
		--ehsmAPIKEY $apikey
	elif [ "$KMS_TYPE" = "simple" ]; then
	    appid=123456654321
	    apikey=123456654321
		java -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
		com.intel.analytics.bigdl.ppml.examples.GeneratePrimaryKey \
		--primaryKeyPath /opt/occlum_spark/data/key/simple_encrypted_primary_key \
		--kmsType SimpleKeyManagementService \
		--simpleAPPID $appid \
		--simpleAPIKEY $apikey
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
		return -1
	fi
elif [ "$action" = "encrypt" ]; then
	appid=$3
	apikey=$4
	input_path=$5
	if [ "$KMS_TYPE" = "ehsm" ]; then
	  /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                    -XX:-UseCompressedOops \
                    -XX:ActiveProcessorCount=4 \
                    -Divy.home="/tmp/.ivy" \
                    -Dos.name="Linux" \
                    -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
                    -Xmx512m org.apache.spark.deploy.SparkSubmit \
                    --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
                    --conf spark.bigdl.primaryKey.BobPK.kms.type=EHSMKeyManagementService \
                    --conf spark.bigdl.primaryKey.BobPK.kms.ip=$EHSM_KMS_IP \
                    --conf spark.bigdl.primaryKey.BobPK.kms.port=$EHSM_KMS_PORT \
                    --conf spark.bigdl.primaryKey.BobPK.kms.appId=$appid \
                    --conf spark.bigdl.primaryKey.BobPK.kms.apiKey=$apikey \
                    --conf spark.bigdl.primaryKey.BobPK.material=/opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
                    --class com.intel.analytics.bigdl.ppml.utils.Encrypt \
                    $SPARK_HOME/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar,$SPARK_HOME/examples/jars/scopt_2.12-3.7.1.jar \
                    --inputDataSourcePath $input_path \
                    --outputDataSinkPath /opt/occlum_spark/data/encryptEhsm/ \
                    --cryptoMode aes/cbc/pkcs5padding \
                    --dataSourceType csv

	elif [ "$KMS_TYPE" = "simple" ]; then
		appid=123456654321
    apikey=123456654321
    input_path=$5
    /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                    -XX:-UseCompressedOops \
                    -XX:ActiveProcessorCount=4 \
                    -Divy.home="/tmp/.ivy" \
                    -Dos.name="Linux" \
                    -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
                    -Xmx512m org.apache.spark.deploy.SparkSubmit \
                    --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
                    --conf spark.bigdl.primaryKey.AmyPK.kms.type=SimpleKeyManagementService \
                    --conf spark.bigdl.primaryKey.AmyPK.kms.appId=$appid \
                    --conf spark.bigdl.primaryKey.AmyPK.kms.apiKey=$apikey \
                    --conf spark.bigdl.primaryKey.AmyPK.material=/opt/occlum_spark/data/key/simple_encrypted_primary_key \
                    --class com.intel.analytics.bigdl.ppml.utils.Encrypt \
                    $SPARK_HOME/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar,$SPARK_HOME/examples/jars/scopt_2.12-3.7.1.jar \
                    --inputDataSourcePath $input_path \
                    --outputDataSinkPath /opt/occlum_spark/data/encryptSimple/ \
                    --cryptoMode aes/cbc/pkcs5padding \
                    --dataSourceType csv
	else
		echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
                return -1
        fi
elif [ "$action" = "encryptwithrepartition" ]; then
  #not test completely
	if [ "$KMS_TYPE" = "ehsm" ]; then
	    appid=$3
            apikey=$4
	    input_path=$5
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
	    appid=123456654321
            apikey=123456654321
	    input_path=$5
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
    appid=$3
    apikey=$4
    input_path=$5
	if [ "$KMS_TYPE" = "ehsm" ]; then
	/usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                        -XX:-UseCompressedOops \
                        -XX:ActiveProcessorCount=4 \
                        -Divy.home="/tmp/.ivy" \
                        -Dos.name="Linux" \
                        -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
                        -Xmx512m org.apache.spark.deploy.SparkSubmit \
                        --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
                        --conf spark.bigdl.primaryKey.BobPK.kms.type=EHSMKeyManagementService \
                        --conf spark.bigdl.primaryKey.BobPK.kms.ip=$EHSM_KMS_IP \
                        --conf spark.bigdl.primaryKey.BobPK.kms.port=$EHSM_KMS_PORT \
                        --conf spark.bigdl.primaryKey.BobPK.kms.appId=$appid \
                        --conf spark.bigdl.primaryKey.BobPK.kms.apiKey=$apikey \
                        --conf spark.bigdl.primaryKey.BobPK.material=/opt/occlum_spark/data/key/ehsm_encrypted_primary_key \
                        --class com.intel.analytics.bigdl.ppml.utils.Encrypt \
                        $SPARK_HOME/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar,$SPARK_HOME/examples/jars/scopt_2.12-3.7.1.jar \
                        --inputDataSourcePath $input_path \
                        --outputDataSinkPath /opt/occlum_spark/data/decryptEhsm/ \
                        --cryptoMode aes/cbc/pkcs5padding \
                        --dataSourceType csv \
                        --action decrypt
	elif [ "$KMS_TYPE" = "simple" ]; then
	    	appid=123456654321
            apikey=123456654321
            input_path=$5
            /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                            -XX:-UseCompressedOops \
                            -XX:ActiveProcessorCount=4 \
                            -Divy.home="/tmp/.ivy" \
                            -Dos.name="Linux" \
                            -cp $BIGDL_HOME/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/* \
                            -Xmx512m org.apache.spark.deploy.SparkSubmit \
                            --conf spark.hadoop.io.compression.codecs="com.intel.analytics.bigdl.ppml.crypto.CryptoCodec" \
                            --conf spark.bigdl.primaryKey.AmyPK.kms.type=SimpleKeyManagementService \
                            --conf spark.bigdl.primaryKey.AmyPK.kms.appId=$appid \
                            --conf spark.bigdl.primaryKey.AmyPK.kms.apiKey=$apikey \
                            --conf spark.bigdl.primaryKey.AmyPK.material=/opt/occlum_spark/data/key/simple_encrypted_primary_key \
                            --class com.intel.analytics.bigdl.ppml.utils.Encrypt \
                            $SPARK_HOME/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar,$SPARK_HOME/examples/jars/scopt_2.12-3.7.1.jar \
                            --inputDataSourcePath $input_path \
                            --outputDataSinkPath /opt/occlum_spark/data/decryptSimple/ \
                            --cryptoMode aes/cbc/pkcs5padding \
                            --dataSourceType csv \
                            --action decrypt
	else
                echo "Wrong KMS_TYPE! KMS_TYPE can be (1) ehsm, (2) simple "
                return -1
        fi
else
	echo "Wrong action! Action can be (1) generatekey, (2) encrypt, (3) decrypt, and (4) encryptwithrepartition."
fi