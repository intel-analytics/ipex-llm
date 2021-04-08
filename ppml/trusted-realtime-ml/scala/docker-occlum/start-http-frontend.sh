#!/bin/bash

set -x

echo "### Launching HTTP Frontend ###"

redis_host=$REDIS_HOST
core_num=$CORE_NUM
redis_secure_password=`openssl rsautl -inkey /opt/password/key.txt -decrypt </opt/password/output.bin`
https_secure_password=`openssl rsautl -inkey /opt/password/key.txt -decrypt </opt/password/output.bin`

java \
    -Xms2g \
    -Xmx8g \
    -XX:ActiveProcessorCount=${core_num} \
    -Dcom.intel.analytics.zoo.shaded.io.netty.tryReflectionSetAccessible=true \
    -Dakka.http.host-connection-pool.max-connections=100 \
    -Dakka.http.host-connection-pool.max-open-requests=128 \
    -Dakka.actor.default-dispatcher.fork-join-executor.parallelism-min=100 \
    -Dakka.actor.default-dispatcher.fork-join-executor.parallelism-max=120 \
    -Dakka.actor.default-dispatcher.fork-join-executor.parallelism-factor=1 \
    -jar /opt/analytics-zoo/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar \
    --redisHost "${redis_host}" \
    --tokensPerSecond 30 \
    --tokenBucketEnabled true \
    --parallelism 30 \
    --httpsEnabled true \
    --httpsKeyStorePath "/opt/keys/keystore.pkcs12" \
    --httpsKeyStoreToken "${https_secure_password}" | tee ./http-frontend-sgx.log
