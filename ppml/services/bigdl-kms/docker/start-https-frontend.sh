#!/bin/bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


set -x

echo "### Launching BigDL KMS HTTPS Frontend ###"

keywhiz_port=$KEYWHIZ_PORT
xmx_size=$XMX_SIZE
core_num=$CORE_NUM
https_key_store_path=/usr/app/server/src/main/resources/dev_and_test_keystore.p12
https_secure_password=$HTTPS_SECURE_PASSWORD # k8s secret

/opt/jdk8/bin/java \
    -Xms2g \
    -Xmx${xmx_size} \
    -XX:ActiveProcessorCount=${core_num} \
    -Dcom.intel.analytics.zoo.shaded.io.netty.tryReflectionSetAccessible=true \
    -Dakka.http.host-connection-pool.max-connections=100 \
    -Dakka.http.host-connection-pool.max-open-requests=128 \
    -Dakka.actor.default-dispatcher.fork-join-executor.parallelism-min=100 \
    -Dakka.actor.default-dispatcher.fork-join-executor.parallelism-max=120 \
    -Dakka.actor.default-dispatcher.fork-join-executor.parallelism-factor=1 \
    -jar /opt/bigdl-kms-frontend.jar \
    --class com.intel.analytics.bigdl.kms.frontend.App \
    --keywhizHost "keywhiz-service" \
    --tokensPerSecond 30 \
    --tokenBucketEnabled true \
    --parallelism 30 \
    --httpsKeyStorePath "${https_key_store_path}" \
    --httpsKeyStoreToken "${https_secure_password}" | tee ./https-frontend.log
