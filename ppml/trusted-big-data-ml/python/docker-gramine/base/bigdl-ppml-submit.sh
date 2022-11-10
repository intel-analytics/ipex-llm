#!/bin/bash
SGX_ENABLED=false
LOG_FILE="bigdl-ppml-submit.log"
application_args=""
input_args=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --master)
      MASTER="$2"
      input_args="$input_args $1 $2"
      shift # past argument
      shift # past value
      ;;
    --deploy-mode)
      DEPLOY_MODE="$2"
      input_args="$input_args $1 $2"
      shift # past argument
      shift # past value
      ;;
    --sgx-enabled)
      SGX_ENABLED="$2"
      shift # past argument
      shift # past value
      ;;
    --driver-memory)
      DRIVER_MEM="$2"
      shift # past argument
      shift # past value
      ;;
    --sgx-driver-jvm-memory)
      SGX_DRIVER_JVM_MEM="$2"
      shift # past argument
      shift # past value
      ;;
    --executor-memory)
      EXECUTOR_MEM="$2"
      shift # past argument
      shift # past value
      ;;
    --sgx-executor-jvm-memory)
      SGX_EXECUTOR_JVM_MEM="$2"
      shift # past argument
      shift # past value
      ;;
    --verbose)
      input_args="$input_args $1"
      shift # past argument
      ;;
    --log-file)
      LOG_FILE="$2"
      shift
      shift
      ;;
    -*|--*)
      input_args="$input_args $1 $2"
      shift
      shift
      ;;
    *)
      application_args="${@}" # save positional arg
      break
      ;;
  esac
done

echo "input_args $input_args"
echo "app_args $application_args"
echo $MASTER

if [ "$MASTER" == k8s* ]; then
  if [ "$DEPLOY_MODE" = "" ]; then
     echo "--deploy-mode should be specified for k8s"
     exit 1
  elif [ "$DRIVER_MEM" = "" ] || [ "$EXECUTOR_MEM" = "" ]; then
     echo "--driver-memory and --executor-memory should be specified for k8s"
     exit 1
  fi
  default_config="--executor-memory $EXECUTOR_MEM \
                  --driver-memory $DRIVER_MEM"
fi

if [ "$SGX_ENABLED" = "true" ]; then
  if [ "$SGX_DRIVER_JVM_MEM" = "" ] || [ "$SGX_EXECUTOR_JVM_MEM" = "" ]; then
    echo "--sgx-driver-jvm-memory and --sgx-executor-jvm-memory must be specified when sgx is enabled"
    exit 1
  else
    sgx_commands="--conf spark.kubernetes.sgx.enabled=$SGX_ENABLED \
        --conf spark.kubernetes.sgx.driver.jvm.mem=$SGX_DRIVER_JVM_MEM \
        --conf spark.kubernetes.sgx.executor.jvm.mem=$SGX_EXECUTOR_JVM_MEM"
  fi
else
  sgx_commands=""
fi

default_config="${default_config} --conf spark.driver.host=$LOCAL_IP \
        --conf spark.driver.port=$RUNTIME_DRIVER_PORT \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
        --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/spark-driver-template.yaml \
        --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
        --conf spark.kubernetes.executor.deleteOnTermination=false"

if [ $secure_password ]; then
   SSL="--conf spark.authenticate=true \
    --conf spark.authenticate.secret=$secure_password \
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret"  \
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret"  \
    --conf spark.authenticate.enableSaslEncryption=true \
    --conf spark.network.crypto.enabled=true  \
    --conf spark.network.crypto.keyLength=128  \
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    --conf spark.io.encryption.enabled=true \
    --conf spark.io.encryption.keySizeBits=128 \
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1 \
    --conf spark.ssl.enabled=true \
    --conf spark.ssl.port=8043 \
    --conf spark.ssl.keyPassword=$secure_password \
    --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks  \
    --conf spark.ssl.keyStorePassword=$secure_password \
    --conf spark.ssl.keyStoreType=JKS \
    --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.trustStorePassword=$secure_password \
    --conf spark.ssl.trustStoreType=JKS"
else
   SSL=""
fi

spark_submit_command="${JAVA_HOME}/bin/java \
        -cp ${BIGDL_HOME}/jars/*:${SPARK_HOME}/conf/:${SPARK_HOME}/jars/* \
        -Xmx${RUNTIME_DRIVER_MEMORY} \
        org.apache.spark.deploy.SparkSubmit \
        $SSL \
        $default_config \
        $sgx_commands"

set -x

spark_submit_command="${spark_submit_command} ${input_args} ${application_args}"
echo "[INFO] spark_submit_command: ${spark_submit_command}"
if [ "$SGX_ENABLED" == "true" ] && [ "$DEPLOY_MODE" != "cluster" ]; then
    echo "[INFO] sgx enabled and convert spark submit command to sgx command"
    ./init.sh
    export sgx_command=${spark_submit_command}
    gramine-sgx bash 2>&1 | tee $LOG_FILE
else
    $spark_submit_command 2>&1 | tee $LOG_FILE
fi
