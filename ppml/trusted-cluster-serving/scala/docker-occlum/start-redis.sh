#!/bin/bash
# set -x

echo "### Launching Redis ###"

cd /opt/redis
./redis-server -p $REDIS_PORT \
    --protected-mode no --maxmemory 10g | tee ./redis-sgx.log
