#!/bin/bash

set -x

if [ -f "/ppml/trusted-big-data-ml/bash.sig" ]; then
    echo "/ppml/trusted-big-data-ml/bash.sig is ready"
else
    echo "/ppml/trusted-big-data-ml/bash.sig is not ready, please generate it through building CustomerImageDockfile"
    exit 1
fi
if [ -c "/dev/sgx/enclave" ]; then
    echo "/dev/sgx/enclave is ready"
elif [ -c "/dev/sgx_enclave" ]; then
    echo "/dev/sgx/enclave not ready, try to link to /dev/sgx_enclave"
    mkdir -p /dev/sgx
    ln -s /dev/sgx_enclave /dev/sgx/enclave
else
    echo "both /dev/sgx/enclave /dev/sgx_enclave are not ready, please check the kernel and driver"
    exit 1
fi

if [ -c "/dev/sgx/provision" ]; then
    echo "/dev/sgx/provision is ready"
elif [ -c "/dev/sgx_provision" ]; then
    echo "/dev/sgx/provision not ready, try to link to /dev/sgx_provision"
    mkdir -p /dev/sgx
    ln -s /dev/sgx_provision /dev/sgx/provision
else
    echo "both /dev/sgx/provision /dev/sgx_provision are not ready, please check the kernel and driver"
    exit 1
fi

ls -al /dev/sgx

/gramine/python/gramine-sgx-get-token -output /ppml/trusted-big-data-ml/bash.token -sig /ppml/trusted-big-data-ml/bash.sig

chmod +x /ppml/trusted-big-data-ml/bash.token

