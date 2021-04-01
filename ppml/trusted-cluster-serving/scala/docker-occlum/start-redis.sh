#!/bin/bash
# set -x

echo "### Launching Redis ###"

cd /opt/redis
./bin/redis-server --port $REDIS_PORT \
    --protected-mode no --maxmemory 10g | tee ./redis-sgx.log
