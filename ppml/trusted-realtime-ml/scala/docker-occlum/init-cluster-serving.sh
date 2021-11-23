#!/bin/bash
set -x

core_num=$CORE_NUM
redis_host=$REDIS_HOST

sed -i "/modelPath:/c \ \ modelPath: /opt/resnet50" config.yaml
sed -i "/coreNumberPerMachine:/c \ \ coreNumberPerMachine: ${core_num}" config.yaml
sed -i "/redisSecureEnabled:/c \ \ redisSecureEnabled: false" config.yaml
sed -i "/recordEncrypted:/c \ \ recordEncrypted: false" config.yaml
sed -i "/redisUrl:/c \ \ redisUrl: ${redis_host}:6379" config.yaml
