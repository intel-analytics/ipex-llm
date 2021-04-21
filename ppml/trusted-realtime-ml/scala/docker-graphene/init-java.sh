#!/bin/bash

set -x

local_ip=$LOCAL_IP
sgx_mem_size=$SGX_MEM_SIZE
flink_home=$FLINK_HOME

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/ppml/trusted-realtime-ml/java G_SGX_SIZE=$sgx_mem_size FLINK_HOME=$flink_home&& \
cd /opt/jdk8/bin && \
ln -s /ppml/trusted-realtime-ml/java/java.sig java.sig && \
ln -s /ppml/trusted-realtime-ml/java/java.manifest.sgx java.manifest.sgx && \
ln -s /ppml/trusted-realtime-ml/java/java.token java.token
