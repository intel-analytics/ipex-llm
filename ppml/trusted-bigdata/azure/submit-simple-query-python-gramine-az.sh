RUNTIME_SPARK_MASTER=
RUNTIME_K8S_SPARK_IMAGE=

DATA_LAKE_NAME=
DATA_LAKE_ACCESS_KEY=
KEY_VAULT_NAME=
PRIMARY_KEY_PATH=/ppml/work/data/keys/primaryKey
DATA_KEY_PATH=/ppml/work/data/keys/dataKey

INPUT_DIR_PATH=abfs://bigdl-data@${DATA_LAKE_NAME}.dfs.core.windows.net/simple-query/input
OUTPUT_DIR_PATH=abfs://bigdl-data@${DATA_LAKE_NAME}.dfs.core.windows.net/simple-query/output

secure_password=`az keyvault secret show --name "key-pass" --vault-name $KEY_VAULT_NAME --query "value" | sed -e 's/^"//' -e 's/"$//'`
echo "secure password is: $secure_password"

export RUNTIME_DRIVER_MEMORY=6g
export RUNTIME_DRIVER_PORT=54321

bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-driver-jvm-memory 2g \
        --sgx-executor-jvm-memory 7g \
        --driver-memory 6g \
	--executor-memory 24g \
        --driver-cores 4 \
        --executor-cores 4 \
        --num-executors 1 \
	--conf spark.cores.max=4 \
    --name simple-query-sgx \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --driver-template /ppml/azure/spark-driver-template-az.yaml \
    --executor-template /ppml/azure/spark-executor-template-az.yaml \
    --key-store /ppml/work/data/keys/keystore.jks \
    --conf spark.kubernetes.file.upload.path=file:///tmp \
    --conf spark.port.maxRetries=100 \
    --conf spark.cores.max=4 \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/*:${SPARK_HOME}/jars/* \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/*:${SPARK_HOME}/jars/* \
    --conf spark.hadoop.fs.azure.account.auth.type.${DATA_LAKE_NAME}.dfs.core.windows.net=SharedKey \
    --conf spark.hadoop.fs.azure.account.key.${DATA_LAKE_NAME}.dfs.core.windows.net=${DATA_LAKE_ACCESS_KEY} \
    --conf spark.hadoop.fs.azure.enable.append.support=true \
    --conf spark.hadoop.hive.server2.enable.doAs=false \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,${BIGDL_HOME}/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip \
     local:///ppml/bigdl-ppml/src/bigdl/ppml/api/simple_query_example.py --primary_key_material /ppml/work/data/primaryKey --vault keyvaultly --input_path $INPUT_DIR_PATH --output_path $OUTPUT_DIR_PATH --input_encrypt_mode AES/CBC/PKCS5Padding --output_encrypt_mode plain_text --kms_type AzureKeyManagementService
