#!/bin/bash

# Attestation
if [ -z "$ATTESTATION" ]; then
  echo "[INFO] Attestation is disabled!"
  ATTESTATION="false"
fi

if [ "$ATTESTATION" ==  "true" ]; then
  if [ -e "/dev/tdx-guest" ]; then
    cd /opt
    bash /opt/attestation.sh
    bash /opt/temp_command_file
    if [ $? -ne 0 ]; then
      echo "[ERROR] Attestation Failed!"
      exit 1
    fi
  else
      echo "TDX device not found!"
  fi
fi

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /usr/bin/tini -s -- "bash"
else
  runtime_command="$@"
  exec $runtime_command
fi
