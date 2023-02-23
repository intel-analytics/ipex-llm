#!/bin/bash

if [ -z $PCCS_VERSION ] ; then
    PCCS_VERSION=v4
fi

# Set PCCS conf
if [ "$PCCS_URL" != "" ] ; then
    echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/'${PCCS_VERSION}'/' > /etc/sgx_default_qcnl.conf
    echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
fi

if [ -z $ATTESTATION_SERVICE_HOST ] ; then
    ATTESTATION_SERVICE_HOST=0.0.0.0
fi
if [ -z $ATTESTATION_SERVICE_PORT ] ; then
    ATTESTATION_SERVICE_PORT=9875
fi

export sgx_command="java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.attestation.BigDLRemoteAttestationService -h $ATTESTATION_SERVICE_HOST -p $ATTESTATION_SERVICE_PORT -t $HTTPS_KEY_STORE_TOKEN -k $SECRET_KEY"

if [ "$SGX_ENABLED" == "true" ] ; then
    ./init.sh
    gramine-sgx bash | tee bigdl-as.log
else 
    $sgx_command | tee bigdl-as.log
fi
