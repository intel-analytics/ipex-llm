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
  ATTESTATION_COMMAND="/opt/jdk8/bin/java -Xmx1g -cp $SPARK_CLASSPATH:$BIGDL_HOME/jars/* com.intel.analytics.bigdl.ppml.attestation.AttestationCLI -u ${ATTESTATION_URL} -i ${ATTESTATION_ID}  -k ${ATTESTATION_KEY}"
fi
