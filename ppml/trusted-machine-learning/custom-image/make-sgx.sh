#!/bin/bash

echo SGX_MEM_SIZE:$SGX_MEM_SIZE
echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL
echo ENABLE_DCAP_ATTESTATION:$ENABLE_DCAP_ATTESTATION
if [[ "$ENABLE_DCAP_ATTESTATION" == "false" ]]; then
   echo "Warning: Disable dcap! Do not use this in production!"
   sed -i 's/"dcap"/"none"/g' bash.manifest.template
fi
cat bash.manifest.template|grep sgx.remote_attestation
make SGX=1 DEBUG=1 THIS_DIR=/ppml/trusted-big-data-ml  SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL
