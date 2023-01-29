#!/bin/bash

# Set PCCS conf
if [ "$PCCS_URL" != "" ] ; then
    echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v4/' > /etc/sgx_default_qcnl.conf
    echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
fi

if [ -z $ATTESTATION_SERVICE_HOST ] ; then
    ATTESTATION_SERVICE_HOST=0.0.0.0
fi
if [ -z $ATTESTATION_SERVICE_PORT ] ; then
    ATTESTATION_SERVICE_PORT=9875
fi
java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.attestation.BigDLRemoteAttestationService -h $ATTESTATION_SERVICE_HOST -p $ATTESTATION_SERVICE_PORT
