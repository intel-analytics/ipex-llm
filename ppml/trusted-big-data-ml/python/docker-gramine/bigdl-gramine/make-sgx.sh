#!/bin/bash

echo SGX_MEM_SIZE:$SGX_MEM_SIZE
echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL
echo ATTESTATION:$ATTESTATION
if [[ "$ATTESTATION" == "false" ]]; then
   sed -i 's/"dcap"/"none"/g' bash.manifest.template
fi
make SGX=1 DEBUG=1 THIS_DIR=/ppml/trusted-big-data-ml  SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL
