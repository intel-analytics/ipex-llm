#!/bin/bash

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
  if [ -z "$APP_ID" ]; then
    echo "[ERROR] Attestation is enabled, but APP_ID is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$API_KEY" ]; then
    echo "[ERROR] Attestation is enabled, but API_KEY is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$REPORT_DATA" ]; then
    echo "[INFO] Attestation is enabled, use default REPORT_DATA ppml"
    export REPORT_DATA="ppml"
  fi
  cd /opt/occlum_spark
  occlum exec /bin/dcap_c_test $REPORT_DATA
  echo "generate quote success"
  ATTESTATION_COMMAND="occlum exec /bin/python3 /opt/attestation_cli.py -u ${ATTESTATION_URL} -i ${APP_ID} -k ${API_KEY} -O Occlum"
  ## default is null
  if [ -n "$ATTESTATION_POLICYID" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -o ${ATTESTATION_POLICYID}"
  fi
  ## default is BIGDL
  if [ -n "$ATTESTATION_TYPE" ]; then
    ATTESTATION_COMMAND="${ATTESTATION_COMMAND} -t ${ATTESTATION_TYPE}"
  fi
  echo $ATTESTATION_COMMAND > temp_command_file
  echo 'if [ $? -gt 0 ]; then ' >> temp_command_file
  echo '  exit 1' >> temp_command_file
  echo 'fi' >> temp_command_file
fi