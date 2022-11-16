#set -x
export RUNTIME_DRIVER_MEMORY=8g

RUNTIME_SPARK_MASTER=
AZ_CONTAINER_REGISTRY=
BIGDL_VERSION=2.1.0
SPARK_EXTRA_JAR_PATH=
SPARK_JOB_MAIN_CLASS=
ARGS=
DATA_LAKE_NAME=
DATA_LAKE_ACCESS_KEY=
KEY_VAULT_NAME=
PRIMARY_KEY_PATH=
DATA_KEY_PATH=

secure_password=`az keyvault secret show --name "key-pass" --vault-name $KEY_VAULT_NAME --query "value" | sed -e 's/^"//' -e 's/"$//'`

bash bigdl-ppml-submit.sh \
	--master $RUNTIME_SPARK_MASTER \
	--deploy-mode client \
	--sgx-enabled true \
	--sgx-log-level error \
	--sgx-driver-memory 16g \
	--sgx-driver-jvm-memory 7g \
	--sgx-executor-memory 16g \
	--sgx-executor-jvm-memory 7g \
	--driver-memory 18g \
	--driver-cores 4 \
	--executor-memory 18g \
	--executor-cores 4 \
	--num-executors 2 \
	--conf spark.cores.max=16 \
    --name spark-decrypt-sgx \
    --conf spark.kubernetes.container.image=$AZ_CONTAINER_REGISTRY.azurecr.io/intel_corporation/bigdl-ppml-trusted-big-data-ml-python-graphene:$BIGDL_VERSION \
    --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/azure/spark-driver-template-az.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/azure/spark-executor-template-az.yaml \
    --jars local://$SPARK_EXTRA_JAR_PATH \
    --conf spark.hadoop.fs.azure.account.auth.type.${DATA_LAKE_NAME}.dfs.core.windows.net=SharedKey \
    --conf spark.hadoop.fs.azure.account.key.${DATA_LAKE_NAME}.dfs.core.windows.net=${DATA_LAKE_ACCESS_KEY} \
    --conf spark.hadoop.fs.azure.enable.append.support=true \
    --conf spark.bigdl.kms.type=AzureKeyManagementService \
    --conf spark.bigdl.kms.azure.vault=$KEY_VAULT_NAME \
    --conf spark.bigdl.kms.key.primary=$PRIMARY_KEY_PATH \
    --conf spark.bigdl.kms.key.data=$DATA_KEY_PATH \
    --class $SPARK_JOB_MAIN_CLASS \
    --verbose \
    $SPARK_EXTRA_JAR_PATH \
    $ARGS
