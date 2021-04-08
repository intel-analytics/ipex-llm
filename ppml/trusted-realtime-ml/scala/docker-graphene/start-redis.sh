#!/bin/bash
# set -x

echo "### Launching Redis ###"

cd /ppml/trusted-cluster-serving/redis
SGX=1 ./pal_loader redis-server --tls-port $REDIS_PORT --port 0 \
    --tls-cert-file /ppml/trusted-cluster-serving/redis/work/keys/server.crt \
    --tls-key-file /ppml/trusted-cluster-serving/redis/work/keys/server.key \
    --tls-ca-cert-file /ppml/trusted-cluster-serving/redis/work/keys/server.crt \
    --protected-mode no --maxmemory 10g | tee ./redis-sgx.log
