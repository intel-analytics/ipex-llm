#!/bin/bash
set -x

export BIGDL_PPML_JAR=$BIGDL_HOME/jars/*

#for production
echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf

if [[ -z "$ATTESTATION_URL" ]]; then
    echo "[ERROR] ATTESTATION_URL is not set!"
    echo "[INFO] PPML Application Exit!"
    exit 1
fi
if [[ -z "$ATTESTATION_TYPE" ]]; then
    ATTESTATION_TYPE="EHSMAttestationService"
fi
if [[ -z "$APP_ID" ]]; then
    echo "[ERROR] APP_ID is not set!"
    echo "[INFO] PPML Application Exit!"
    exit 1
fi
if [[ -z "$API_KEY" ]]; then
    echo "[ERROR] API_KEY is not set!"
    echo "[INFO] PPML Application Exit!"
    exit 1
fi
if [[ -z "$CHALLENGE" ]]; then
    #echo ppmltest|base64
    CHALLENGE=cHBtbHRlc3QK
fi
if [[ -z "$SPARK_HOME" ]]; then
    echo "[ERROR] SPARK_HOME is not set!"
    echo "[INFO] PPML Application Exit!"
    exit 1
fi
if [[ -z "$BIGDL_PPML_JAR" ]]; then
    echo "[ERROR] BIGDL_PPML_JAR is not set!"
    echo "[INFO] PPML Application Exit!"
    exit 1
fi

JARS="$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_PPML_JAR"

java -cp $JARS com.intel.analytics.bigdl.ppml.attestation.VerificationCLI -i $APP_ID -k $API_KEY -c $CHALLENGE -u $ATTESTATION_URL -t $ATTESTATION_TYPE
