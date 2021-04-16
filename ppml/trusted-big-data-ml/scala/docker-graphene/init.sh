#!/bin/bash

set -x

local_ip=$LOCAL_IP
sgx_mem_size=$SGX_MEM_SIZE

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/ppml/trusted-big-data-ml SPARK_LOCAL_IP=$local_ip SPARK_USER=root G_SGX_SIZE=$sgx_mem_size && \
cd /opt/jdk8/bin && \
ln -s /ppml/trusted-big-data-ml/java.sig java.sig && \
ln -s /ppml/trusted-big-data-ml/java.manifest.sgx java.manifest.sgx && \
ln -s /ppml/trusted-big-data-ml/java.token java.token
