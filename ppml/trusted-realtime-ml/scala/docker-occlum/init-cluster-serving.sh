#!/bin/bash
set -x

core_num=$CORE_NUM
secure_password=`openssl rsautl -inkey /opt/password/key.txt -decrypt </opt/password/output.bin`
redis_host=$REDIS_HOST

sed -i "/modelPath:/c \ \ modelPath: /opt/resnet50" config.yaml
sed -i "/coreNumberPerMachine:/c \ \ coreNumberPerMachine: ${core_num}" config.yaml
sed -i "/redisSecureEnabled:/c \ \ redisSecureEnabled: true" config.yaml
sed -i "/redisSecureTrustStorePath:/c \ \ redisSecureTrustStorePath: /opt/redis/work/keys/keystore.jks" config.yaml
sed -i "/redisSecureTrustStorePassword:/c \ \ redisSecureTrustStorePassword: ${secure_password}" config.yaml
sed -i "/redisUrl:/c \ \ redisUrl: ${redis_host}:6379" config.yaml
