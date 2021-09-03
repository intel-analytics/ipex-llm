#!/bin/bash
# set -x
sgx_mode=$SGX_MODE

if [[ $sgx_mode == "sgx" || $sgx_mode == "SGX" ]];then cmd_prefix="graphene-sgx ./"; fi

echo "### Launching Redis ###"

cd /ppml/trusted-realtime-ml/redis
eval ${cmd_prefix}bash -c "  ./redis-server --tls-port $REDIS_PORT --port 0 \
    --tls-cert-file /ppml/trusted-realtime-ml/redis/work/keys/server.crt \
    --tls-key-file /ppml/trusted-realtime-ml/redis/work/keys/server.key \
    --tls-ca-cert-file /ppml/trusted-realtime-ml/redis/work/keys/server.crt \
    --protected-mode no --maxmemory 10g " | tee ./redis-${sgx_mode}.log
