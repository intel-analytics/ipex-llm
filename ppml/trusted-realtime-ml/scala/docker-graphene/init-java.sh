#!/bin/bash

set -x

local_ip=$LOCAL_IP
sgx_mem_size=$SGX_MEM_SIZE
flink_home=$FLINK_HOME

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/ppml/trusted-cluster-serving/java G_SGX_SIZE=$sgx_mem_size FLINK_HOME=$flink_home&& \
cd /opt/jdk8/bin && \
ln -s /ppml/trusted-cluster-serving/java/java.sig java.sig && \
ln -s /ppml/trusted-cluster-serving/java/java.manifest.sgx java.manifest.sgx && \
ln -s /ppml/trusted-cluster-serving/java/java.token java.token
