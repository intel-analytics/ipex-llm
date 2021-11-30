#!/bin/bash

set -x

local_ip=$LOCAL_IP
sgx_mem_size=$SGX_MEM_SIZE

if [ -c "/dev/sgx/enclave" ]; then
    echo "/dev/sgx/enclave is ready"
elif [ -c "/dev/sgx_enclave" ]; then
    echo "/dev/sgx/enclave not ready, try to link to /dev/sgx_enclave"
    mkdir -p /dev/sgx
    ln -s /dev/sgx_enclave /dev/sgx/enclave
else
    echo "both /dev/sgx/enclave /dev/sgx_enclave are not ready, please check the kernel and driver"
fi

if [ -c "/dev/sgx/provision" ]; then
    echo "/dev/sgx/provision is ready"
elif [ -c "/dev/sgx_provision" ]; then
    echo "/dev/sgx/provision not ready, try to link to /dev/sgx_provision"
    mkdir -p /dev/sgx
    ln -s /dev/sgx_provision /dev/sgx/provision
else
    echo "both /dev/sgx/provision /dev/sgx_provision are not ready, please check the kernel and driver"
fi

ls -al /dev/sgx

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/ppml/trusted-big-data-ml  SPARK_LOCAL_IP=$local_ip SPARK_USER=root G_SGX_SIZE=$sgx_mem_size
