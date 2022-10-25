#!/bin/bash
set -x

export ATTESTATION_URL=your_attestation_url
export ATTESTATION_TYPE=your_attestation_type
export ATTESTATION_ID=your_app_id
export ATTESTATION_KEY=your_api_key
export ATTESTATION_CHALLENGE=your_attestaion_challege
export ATTESTATION_POLICYID=your_policy_id
export ATTESTATION=true

# Attestation
if [ -z "$ATTESTATION" ]; then
    echo "[INFO] Attestation is disabled!"
    ATTESTATION="false"
elif [ "$ATTESTATION" = "true" ]; then
  echo "[INFO] Attestation is enabled!"
  # Build ATTESTATION_COMMAND
  if [ -z "$ATTESTATION_URL" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_URL is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$ATTESTATION_ID" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_ID is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$ATTESTATION_KEY" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_KEY is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  ATTESTATION_COMMAND="java -Xmx1g -cp /ppml/jars/*: com.intel.analytics.bigdl.ppml.attestation.AttestationCLI -u ${ATTESTATION_URL} -i ${ATTESTATION_ID}  -k ${ATTESTATION_KEY}"
  if [ -n "$ATTESTATION_CHALLENGE" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -c ${ATTESTATION_CHALLENGE}"
  fi
  if [ -n "$ATTESTATION_POLICYID" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -o ${ATTESTATION_POLICYID}"
  fi
fi

export sgx_command=$ATTESTATION_COMMAND
echo $sgx_command
mkdir -p /root/.config/gramine
openssl genrsa -3 -out /root/.config/gramine/enclave-key.pem 3072
gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs
make SGX=1 DEBUG=1 THIS_DIR=/ppml SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=error
bash init.sh
gramine-sgx bash
