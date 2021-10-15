#!/bin/bash

DRIVER_MEMORY=10G
# find driver memory in spark-defaults.conf
if [ -f $SPARK_HOME/conf/spark-defaults.conf ]; then
  mem=`cat $SPARK_HOME/conf/spark-defaults.conf | grep "spark.driver.memory"`
  arrMem=(${mem// / })
  if [ "${arrMem[0]}" == "spark.driver.memory" ]
    then DRIVER_MEMORY="${arrMem[1]}"
  fi
fi
# find driver memory in parameters
for param in "$@"
do
  if [ $DRIVER_MEMORY == 'next' ]
    then DRIVER_MEMORY=$param
  fi
  if [ $param == "--driver-memory" ]
    then DRIVER_MEMORY='next'
  fi
done

if [ $secure_password ]; then
   SSL="-Dspark.authenticate=true \
    -Dspark.authenticate.secret=$secure_password \
    -Dspark.network.crypto.enabled=true \
    -Dspark.network.crypto.keyLength=128 \
    -Dspark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    -Dspark.io.encryption.enabled=true \
    -Dspark.io.encryption.keySizeBits=128 \
    -Dspark.io.encryption.keygen.algorithm=HmacSHA1 \
    -Dspark.ssl.enabled=true \
    -Dspark.ssl.port=8043 \
    -Dspark.ssl.keyPassword=$secure_password \
    -Dspark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    -Dspark.ssl.keyStorePassword=$secure_password \
    -Dspark.ssl.keyStoreType=JKS \
    -Dspark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    -Dspark.ssl.trustStorePassword=$secure_password \
    -Dspark.ssl.trustStoreType=JKS"
else 
   SSL=""
fi

set -x

SGX=1 ./pal_loader bash -c "${JAVA_HOME}/bin/java $SSL \
        -XX:ActiveProcessorCount=24 \
        -cp "/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*" \
        -Xmx${DRIVER_MEMORY} \
        org.apache.spark.deploy.SparkSubmit "$@""
